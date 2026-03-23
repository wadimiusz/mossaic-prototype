"""Train tiny transformers to simulate deterministic finite automata (DFAs).

Task: given a sequence of input symbols, predict the current DFA state.
The DFA is randomly generated with known transition structure, so the
ground truth (which states the model should encode) is fully specified.

Produces: dfa_models/ directory with one .pt file per model.
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
OUTPUT_DIR = "dfa_models"

CONFIGS = [
    {"n_states": 3, "n_symbols": 2, "seq_len": 6},
    {"n_states": 4, "n_symbols": 2, "seq_len": 8},
    {"n_states": 5, "n_symbols": 3, "seq_len": 6},
    {"n_states": 6, "n_symbols": 2, "seq_len": 8},
]
SEEDS = [42, 123, 456]


# ---- DFA generation ----


def make_random_dfa(n_states, n_symbols, rng):
    """Generate a random DFA. Returns transition[state][symbol] -> next_state."""
    return rng.integers(0, n_states, size=(n_states, n_symbols))


def simulate_dfa(transitions, symbols, start_state=0):
    """Simulate DFA on a sequence, return final state."""
    state = start_state
    for s in symbols:
        state = transitions[state, s]
    return state


# ---- Model ----


class DFATransformer(nn.Module):
    def __init__(
        self, n_symbols, n_states, d_model, nhead, num_layers, dim_ff, seq_len
    ):
        super().__init__()
        self.d_model = d_model
        self.n_states = n_states
        # Tokens: 0..n_symbols-1 for input, n_symbols for BOS
        self.token_emb = nn.Embedding(n_symbols + 1, d_model)
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)  # +1 for BOS
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, n_states)

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


def make_dataset(transitions, n_symbols, seq_len, rng, n_samples=3000):
    """Generate random input sequences and their DFA final states."""
    bos_token = n_symbols
    data = []
    labels = []

    for _ in range(n_samples):
        symbols = rng.integers(0, n_symbols, size=seq_len)
        final_state = simulate_dfa(transitions, symbols)
        seq = [bos_token] + symbols.tolist()
        data.append(seq)
        labels.append(final_state)

    return torch.tensor(data), torch.tensor(labels)


# ---- Training ----


def train_one(cfg, seed):
    n_states = cfg["n_states"]
    n_symbols = cfg["n_symbols"]
    seq_len = cfg["seq_len"]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(
        f"  Training: states={n_states}, symbols={n_symbols}, len={seq_len}, seed={seed}"
    )

    transitions = make_random_dfa(n_states, n_symbols, rng)
    print(f"    Transitions: {transitions.tolist()}")

    model = DFATransformer(
        n_symbols, n_states, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, seq_len
    ).to(DEVICE)

    x_all, y_all = make_dataset(transitions, n_symbols, seq_len, rng, n_samples=3000)
    x_all, y_all = x_all.to(DEVICE), y_all.to(DEVICE)
    num_samples = len(x_all)

    state_dist = np.bincount(y_all.cpu().numpy(), minlength=n_states)
    print(f"    Dataset: {num_samples} seqs, state dist: {state_dist.tolist()}")

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
        print(f"    WARNING: acc={acc:.3f}")

    with torch.no_grad():
        residuals = model.get_residual_stream(x_all).cpu().numpy()

    config = {
        "task": "dfa",
        "n_states": n_states,
        "n_symbols": n_symbols,
        "seq_len": seq_len,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF,
        "seed": seed,
    }
    weights_np = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    name = f"dfa_s{n_states}_a{n_symbols}_l{seq_len}_seed{seed}"
    save_path = os.path.join(OUTPUT_DIR, f"{name}.pt")
    torch.save(
        {
            "config": config,
            "weights_np": weights_np,
            "residuals": residuals,
            "inputs": x_all.cpu().numpy(),
            "labels": y_all.cpu().numpy(),
            "transitions": transitions,
        },
        save_path,
    )
    print(f"    Saved to {save_path}")
    return acc


def _train_worker(args):
    cfg, seed = args
    acc = train_one(cfg, seed)
    return (cfg["n_states"], cfg["n_symbols"], cfg["seq_len"], seed, acc)


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
            ns, na, sl, seed, acc = result
            status = "OK" if acc > 0.95 else "LOW"
            print(
                f"  [{done}/{len(jobs)}] s={ns} a={na} l={sl} seed={seed}: acc={acc:.4f} [{status}]"
            )

    print(f"\n{'=' * 50}")
    print("Summary:")
    print("=" * 50)
    results.sort()
    for ns, na, sl, seed, acc in results:
        status = "OK" if acc > 0.95 else "FAILED"
        print(
            f"  states={ns}, symbols={na}, len={sl}, seed={seed}: acc={acc:.4f}  [{status}]"
        )


if __name__ == "__main__":
    main()
