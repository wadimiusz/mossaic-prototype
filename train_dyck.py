"""Train tiny transformers on Dyck-1 language (balanced parentheses).

Task: given a prefix of parentheses, predict the current nesting depth.
The hidden state should encode depth — an integer counter — which is a
fundamentally different structure from Fourier (circular) or ordering (linear).

Tokens: 0="(", 1=")", 2="<BOS>", 3="<PAD>"

Produces: dyck_models/ directory with one .pt file per model.
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
OUTPUT_DIR = "dyck_models"

CONFIGS = [
    {"max_depth": 3, "max_len": 8},
    {"max_depth": 4, "max_len": 10},
    {"max_depth": 5, "max_len": 12},
    {"max_depth": 6, "max_len": 14},
]
SEEDS = [42, 123, 456]

TOK_OPEN = 0
TOK_CLOSE = 1
TOK_BOS = 2
TOK_PAD = 3
VOCAB_SIZE = 4


# ---- Model ----


class DyckTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_ff, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def get_residual_stream(self, x, padding_mask=None):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return h


# ---- Data generation ----


def generate_all_prefixes(max_depth, max_len):
    """Generate all valid Dyck-1 prefixes up to max_len with their depths."""
    results = []

    def recurse(prefix, depth):
        seq = [TOK_BOS] + prefix
        results.append((seq, depth))

        if len(seq) >= max_len:
            return
        if depth < max_depth:
            recurse(prefix + [TOK_OPEN], depth + 1)
        if depth > 0:
            recurse(prefix + [TOK_CLOSE], depth - 1)

    recurse([], 0)
    return results


def make_dataset(max_depth, max_len):
    """Generate training data: predict depth given prefix."""
    prefixes = generate_all_prefixes(max_depth, max_len)

    data = []
    labels = []
    for seq, depth in prefixes:
        padded = seq + [TOK_PAD] * (max_len - len(seq))
        data.append(padded)
        labels.append(depth)

    return (
        torch.tensor(data),
        torch.tensor(labels),
        np.array(labels),
    )


# ---- Training ----


def train_one(cfg, seed):
    max_depth = cfg["max_depth"]
    max_len = cfg["max_len"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Training: depth={max_depth}, len={max_len}, seed={seed}")

    model = DyckTransformer(D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, max_len).to(DEVICE)

    x_all, y_all, depths = make_dataset(max_depth, max_len)
    x_all, y_all = x_all.to(DEVICE), y_all.to(DEVICE)
    num_samples = len(x_all)

    if num_samples == 0:
        print("    WARNING: no data!")
        return 0.0

    print(
        f"    Dataset: {num_samples} prefixes, depth dist: {np.bincount(depths).tolist()}"
    )

    num_classes = max_depth + 1
    depth_head = nn.Linear(D_MODEL, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(depth_head.parameters()), lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        depth_head.train()
        perm = torch.randperm(num_samples, device=DEVICE)
        x_shuf, y_shuf = x_all[perm], y_all[perm]

        total_loss = 0
        correct = 0
        for i in range(0, num_samples, BATCH_SIZE):
            xb = x_shuf[i : i + BATCH_SIZE]
            yb = y_shuf[i : i + BATCH_SIZE]
            padding_mask = xb == TOK_PAD
            h = model.get_residual_stream(xb, padding_mask=padding_mask)
            lengths = (xb != TOK_PAD).sum(dim=1) - 1
            last_h = h[torch.arange(len(xb), device=DEVICE), lengths]
            logits = depth_head(last_h)
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
    depth_head.eval()
    with torch.no_grad():
        padding_mask = x_all == TOK_PAD
        h = model.get_residual_stream(x_all, padding_mask=padding_mask)
        lengths = (x_all != TOK_PAD).sum(dim=1) - 1
        last_h = h[torch.arange(num_samples, device=DEVICE), lengths]
        logits = depth_head(last_h)
        acc = (logits.argmax(1) == y_all).float().mean().item()
    print(f"    Final accuracy: {acc:.4f}")

    if acc < 0.95:
        print(f"    WARNING: acc={acc:.3f}")

    residuals = last_h.cpu().numpy()

    config = {
        "task": "dyck",
        "max_depth": max_depth,
        "max_len": max_len,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF,
        "num_classes": num_classes,
        "vocab_size": VOCAB_SIZE,
        "seed": seed,
    }
    weights_np = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    weights_np["depth_head.weight"] = depth_head.weight.cpu().detach().numpy()
    weights_np["depth_head.bias"] = depth_head.bias.cpu().detach().numpy()

    name = f"dyck_d{max_depth}_l{max_len}_seed{seed}"
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
    return (cfg["max_depth"], cfg["max_len"], seed, acc)


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
            md, ml, seed, acc = result
            status = "OK" if acc > 0.95 else "LOW"
            print(
                f"  [{done}/{len(jobs)}] d={md} l={ml} seed={seed}: acc={acc:.4f} [{status}]"
            )

    print(f"\n{'=' * 50}")
    print("Summary:")
    print("=" * 50)
    results.sort()
    for md, ml, seed, acc in results:
        status = "OK" if acc > 0.95 else "FAILED"
        print(f"  depth={md}, len={ml}, seed={seed}: acc={acc:.4f}  [{status}]")


if __name__ == "__main__":
    main()
