"""Train tiny transformers on modular addition (a + b mod P).

Trains multiple models across different primes and random seeds.
Each model is saved with its weights, config, and hidden-state samples.

Produces: modular_addition_models/ directory with one .pt file per model.
"""

import os

import numpy as np
import torch
import torch.nn as nn

# ---- Config ----

D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DIM_FF = 128
MAX_LEN = 3  # [a, b, =] -> predict (a+b) mod P
BATCH_SIZE = 256
EPOCHS = 500
LR = 3e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "modular_addition_models"

# Multiple primes and seeds for generalization testing
PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
SEEDS = [42, 123, 456]


# ---- Model ----


class ModAddTransformer(nn.Module):
    def __init__(self, p, d_model, nhead, num_layers, dim_ff):
        super().__init__()
        self.p = p
        self.d_model = d_model
        # Tokens: 0..P-1 for numbers, P for "=" separator
        self.token_emb = nn.Embedding(p + 1, d_model)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Weight-tied head: unembedding = embedding[:P].T
        # No separate head parameters — logits = residual @ token_emb[:P].T
        self.head_bias = nn.Parameter(torch.zeros(p))

    def forward(self, x):
        # x: (batch, 3) -- [a, b, =]
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        # Weight-tied: logits = residual @ embedding.T + bias
        return h[:, -1] @ self.token_emb.weight[: self.p].T + self.head_bias

    def get_residual_stream(self, x):
        """Extract the residual stream at the last position after the encoder."""
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return h[:, -1]  # (batch, d_model)


# ---- Data generation ----


def make_dataset(p):
    """Generate all P*P examples for modular addition."""
    data = []
    labels = []
    eq_token = p  # "=" token
    for a in range(p):
        for b in range(p):
            data.append([a, b, eq_token])
            labels.append((a + b) % p)
    return torch.tensor(data), torch.tensor(labels)


# ---- Training ----


def train_one(p, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 50}")
    print(f"Training: mod {p}, seed {seed}")
    print("=" * 50)

    model = ModAddTransformer(p, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF).to(DEVICE)
    param_count = sum(par.numel() for par in model.parameters())
    print(f"  Parameters: {param_count:,}")

    x_all, y_all = make_dataset(p)
    x_all, y_all = x_all.to(DEVICE), y_all.to(DEVICE)
    n = len(x_all)
    print(f"  Dataset: {n} examples (all {p}x{p} pairs)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        # Shuffle
        perm = torch.randperm(n, device=DEVICE)
        x_shuf, y_shuf = x_all[perm], y_all[perm]

        total_loss = 0
        correct = 0
        for i in range(0, n, BATCH_SIZE):
            xb = x_shuf[i : i + BATCH_SIZE]
            yb = y_shuf[i : i + BATCH_SIZE]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            acc = correct / n
            print(
                f"  Epoch {epoch + 1:3d}/{EPOCHS}: loss={total_loss / n:.4f}  acc={acc:.3f}"
            )

    # Final accuracy
    model.eval()
    with torch.no_grad():
        logits = model(x_all)
        acc = (logits.argmax(1) == y_all).float().mean().item()
    print(f"  Final accuracy: {acc:.4f}")

    if acc < 0.95:
        print(f"  WARNING: model did not learn the task well (acc={acc:.3f})")

    # Extract residual stream for all inputs
    with torch.no_grad():
        residuals = model.get_residual_stream(x_all).cpu().numpy()

    # Save
    config = {
        "p": p,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF,
        "max_len": MAX_LEN,
        "seed": seed,
    }
    weights_np = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    save_path = os.path.join(OUTPUT_DIR, f"mod{p}_seed{seed}.pt")
    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
            "weights_np": weights_np,
            "residuals": residuals,  # (P*P, d_model)
            "inputs": x_all.cpu().numpy(),
            "labels": y_all.cpu().numpy(),
        },
        save_path,
    )
    print(f"  Saved to {save_path}")
    return acc


def _train_worker(args):
    """Worker for parallel training."""
    p, seed = args
    acc = train_one(p, seed)
    return (p, seed, acc)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Primes: {PRIMES}")
    print(f"Seeds: {SEEDS}")

    jobs = [(p, seed) for p in PRIMES for seed in SEEDS]

    # Train in parallel using multiprocessing (GPU handles the tiny models fine)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

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
            p, seed, acc = result
            status = "OK" if acc > 0.95 else "LOW"
            print(
                f"  [{done}/{len(jobs)}] mod {p:2d} seed {seed:3d}: acc={acc:.4f} [{status}]"
            )

    print(f"\n{'=' * 50}")
    print("Summary:")
    print("=" * 50)
    for p, seed, acc in results:
        status = "OK" if acc > 0.95 else "FAILED"
        print(f"  mod {p:2d}, seed {seed:3d}: acc={acc:.4f}  [{status}]")


if __name__ == "__main__":
    main()
