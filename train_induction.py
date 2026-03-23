"""Train tiny transformers on induction head pattern completion.

Task: given [..., A, B, ..., A, ?], predict B.
Requires attention-based pattern matching (the "induction head" motif).

Produces: induction_models/ directory with one .pt file per model.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn

# ---- Config ----

D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DIM_FF = 128
BATCH_SIZE = 256
EPOCHS = 500
LR = 3e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "induction_models"

CONFIGS = [
    {"vocab_size": 8, "seq_len": 8},
    {"vocab_size": 12, "seq_len": 10},
    {"vocab_size": 16, "seq_len": 12},
    {"vocab_size": 20, "seq_len": 14},
]
SEEDS = [42, 123, 456]


# ---- Model ----


class InductionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_ff, seq_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)  # +1 for "?" query token
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        return self.head(h[:, -1])

    def get_residual_stream(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        return h[:, -1]


# ---- Data generation ----


def make_induction_dataset(vocab_size, seq_len, rng, n_samples=3000):
    """Generate induction examples: [..., A, B, ..., A, ?] -> B."""
    query_token = vocab_size
    data = []
    labels = []

    for _ in range(n_samples):
        A = rng.integers(0, vocab_size)
        B = rng.integers(0, vocab_size)

        # First [A, B] in the first half
        first_pos = rng.integers(0, max(1, seq_len // 2 - 1))
        # Second A near the end (before query position)
        second_pos = rng.integers(max(first_pos + 2, seq_len // 2), seq_len - 1)

        # Fill with random tokens, avoid accidental A's
        seq = rng.integers(0, vocab_size, size=seq_len)
        for i in range(seq_len - 1):
            if seq[i] == A and i != first_pos and i != second_pos:
                seq[i] = (A + 1) % vocab_size

        seq[first_pos] = A
        if first_pos + 1 < seq_len:
            seq[first_pos + 1] = B
        seq[second_pos] = A
        seq[seq_len - 1] = query_token

        data.append(seq.tolist())
        labels.append(B)

    return torch.tensor(data), torch.tensor(labels)


# ---- Training ----


def train_one(cfg, seed):
    vocab_size = cfg["vocab_size"]
    seq_len = cfg["seq_len"]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"  Training: vocab={vocab_size}, len={seq_len}, seed={seed}")

    model = InductionTransformer(
        vocab_size, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, seq_len
    ).to(DEVICE)

    x_all, y_all = make_induction_dataset(vocab_size, seq_len, rng, n_samples=3000)
    x_all, y_all = x_all.to(DEVICE), y_all.to(DEVICE)
    num_samples = len(x_all)
    print(f"    Dataset: {num_samples} sequences")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(num_samples, device=DEVICE)
        x_shuf, y_shuf = x_all[perm], y_all[perm]

        total_loss = 0
        correct = 0
        for i in range(0, num_samples, BATCH_SIZE):
            xb = x_shuf[i : i + BATCH_SIZE]
            yb = y_shuf[i : i + BATCH_SIZE]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        if (epoch + 1) % 100 == 0:
            acc = correct / num_samples
            print(
                f"    Epoch {epoch + 1:3d}/{EPOCHS}: loss={total_loss / num_samples:.4f}  acc={acc:.3f}"
            )

    model.eval()
    with torch.no_grad():
        logits = model(x_all)
        acc = (logits.argmax(1) == y_all).float().mean().item()
    print(f"    Final accuracy: {acc:.4f}")

    if acc < 0.90:
        print(f"    WARNING: acc={acc:.3f}")

    with torch.no_grad():
        residuals = model.get_residual_stream(x_all).cpu().numpy()

    config = {
        "task": "induction",
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF,
        "seed": seed,
    }
    weights_np = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    name = f"induction_v{vocab_size}_l{seq_len}_seed{seed}"
    save_path = os.path.join(OUTPUT_DIR, f"{name}.pt")
    torch.save(
        {
            "config": config,
            "weights_np": weights_np,
            "residuals": residuals,
            "inputs": x_all.cpu().numpy(),
            "labels": y_all.cpu().numpy(),
        },
        save_path,
    )
    print(f"    Saved to {save_path}")
    return acc


def _train_worker(args):
    cfg, seed = args
    acc = train_one(cfg, seed)
    return (cfg["vocab_size"], cfg["seq_len"], seed, acc)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")

    jobs = [(cfg, seed) for cfg in CONFIGS for seed in SEEDS]

    mp.set_start_method("spawn", force=True)
    n_workers = min(4, len(jobs))
    print(f"Training {len(jobs)} models with {n_workers} workers")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_train_worker, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            vs, sl, seed, acc = result
            status = "OK" if acc > 0.90 else "LOW"
            print(
                f"  [{done}/{len(jobs)}] v={vs} l={sl} seed={seed}: acc={acc:.4f} [{status}]"
            )

    print(f"\n{'=' * 50}")
    print("Summary:")
    print("=" * 50)
    results.sort()
    for vs, sl, seed, acc in results:
        status = "OK" if acc > 0.90 else "FAILED"
        print(f"  vocab={vs}, len={sl}, seed={seed}: acc={acc:.4f}  [{status}]")


if __name__ == "__main__":
    main()
