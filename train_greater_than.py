"""Train tiny transformers on greater-than comparison (a > b).

Same architecture as modular addition but different task:
given inputs (a, b), predict 1 if a > b, 0 if a <= b.

Produces: greater_than_models/ directory with one .pt file per model.
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
MAX_LEN = 3  # [a, b, ?]
BATCH_SIZE = 256
EPOCHS = 500
LR = 3e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "greater_than_models"

# Different number ranges and seeds
NUM_RANGES = [8, 12, 16, 20, 24]
SEEDS = [42, 123, 456]


# ---- Model ----


class GreaterThanTransformer(nn.Module):
    def __init__(self, n, d_model, nhead, num_layers, dim_ff):
        super().__init__()
        self.n = n
        self.d_model = d_model
        # Tokens: 0..N-1 for numbers, N for "?" separator
        self.token_emb = nn.Embedding(n + 1, d_model)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Binary classification head
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.head(h[:, -1])

    def get_residual_stream(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return h[:, -1]


# ---- Data generation ----


def make_dataset(n):
    """Generate all N*N examples for greater-than comparison."""
    data = []
    labels = []
    sep_token = n  # "?" token
    for a in range(n):
        for b in range(n):
            data.append([a, b, sep_token])
            labels.append(1 if a > b else 0)
    return torch.tensor(data), torch.tensor(labels)


# ---- Training ----


def train_one(n, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Training: N={n}, seed={seed}")

    model = GreaterThanTransformer(n, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF).to(DEVICE)

    x_all, y_all = make_dataset(n)
    x_all, y_all = x_all.to(DEVICE), y_all.to(DEVICE)
    num_samples = len(x_all)

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

    if acc < 0.95:
        print(f"    WARNING: model did not learn the task well (acc={acc:.3f})")

    with torch.no_grad():
        residuals = model.get_residual_stream(x_all).cpu().numpy()

    config = {
        "task": "greater_than",
        "n": n,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF,
        "max_len": MAX_LEN,
        "seed": seed,
    }
    weights_np = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    save_path = os.path.join(OUTPUT_DIR, f"gt_n{n}_seed{seed}.pt")
    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
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
    n, seed = args
    acc = train_one(n, seed)
    return (n, seed, acc)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Number ranges: {NUM_RANGES}")
    print(f"Seeds: {SEEDS}")

    jobs = [(n, seed) for n in NUM_RANGES for seed in SEEDS]

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
            n, seed, acc = result
            status = "OK" if acc > 0.95 else "LOW"
            print(
                f"  [{done}/{len(jobs)}] N={n:2d} seed={seed:3d}: acc={acc:.4f} [{status}]"
            )

    print(f"\n{'=' * 50}")
    print("Summary:")
    print("=" * 50)
    results.sort()
    for n, seed, acc in results:
        status = "OK" if acc > 0.95 else "FAILED"
        print(f"  N={n:2d}, seed={seed:3d}: acc={acc:.4f}  [{status}]")


if __name__ == "__main__":
    main()
