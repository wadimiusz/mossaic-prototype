"""Train a tiny transformer on sentiment classification (Rotten Tomatoes).

Produces:
  toy_transformer.pt  — model weights + tokenizer vocab + config
"""

import random
import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

# ---- Config ----

D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DIM_FF = 128
MAX_LEN = 64
VOCAB_SIZE = 4000  # will be clipped to actual vocab
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "toy_transformer.pt"


# ---- Simple word-level tokenizer ----


def build_vocab(texts, max_vocab):
    """Build word-level vocabulary from texts."""
    counts = Counter()
    for text in texts:
        counts.update(re.findall(r"\w+", text.lower()))
    # Special tokens: 0=PAD, 1=UNK, 2=CLS
    vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
    for word, _ in counts.most_common(max_vocab - len(vocab)):
        vocab[word] = len(vocab)
    return vocab


def tokenize(text, vocab, max_len):
    """Tokenize text to integer indices with CLS prefix."""
    words = re.findall(r"\w+", text.lower())
    ids = [vocab["<CLS>"]] + [vocab.get(w, vocab["<UNK>"]) for w in words]
    ids = ids[:max_len]
    return ids


def collate(batch, vocab, max_len):
    """Pad a batch of (text, label) to uniform length."""
    ids_list = [tokenize(text, vocab, max_len) for text, _ in batch]
    labels = [label for _, label in batch]
    max_seq = max(len(ids) for ids in ids_list)
    padded = [ids + [vocab["<PAD>"]] * (max_seq - len(ids)) for ids in ids_list]
    return torch.tensor(padded), torch.tensor(labels)


# ---- Model ----


class ToyTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, num_layers, dim_ff, max_len, num_classes
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        mask = x == 0  # padding mask
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        # Use CLS token (first position)
        cls = h[:, 0]
        return self.classifier(cls)


# ---- Training ----


def main():
    print(f"Device: {DEVICE}")

    # Load Rotten Tomatoes dataset
    print("Loading dataset...")
    ds = load_dataset("rotten_tomatoes")
    train_data = [(x["text"], x["label"]) for x in ds["train"]]
    val_data = [(x["text"], x["label"]) for x in ds["validation"]]
    test_data = [(x["text"], x["label"]) for x in ds["test"]]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Build vocabulary from train set
    vocab = build_vocab([t for t, _ in train_data], VOCAB_SIZE)
    actual_vocab_size = len(vocab)
    print(f"  Vocab size: {actual_vocab_size}")

    # Create model
    model = ToyTransformer(
        actual_vocab_size, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, MAX_LEN, NUM_CLASSES
    ).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(train_data)

        total_loss = 0
        correct = 0
        total = 0
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i : i + BATCH_SIZE]
            x, y = collate(batch, vocab, MAX_LEN)
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(batch)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_data), BATCH_SIZE):
                batch = val_data[i : i + BATCH_SIZE]
                x, y = collate(batch, vocab, MAX_LEN)
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += len(batch)
        val_acc = val_correct / val_total

        print(
            f"  Epoch {epoch + 1:2d}/{EPOCHS}: "
            f"loss={total_loss / total:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
        )

    # Test accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for i in range(0, len(test_data), BATCH_SIZE):
            batch = test_data[i : i + BATCH_SIZE]
            x, y = collate(batch, vocab, MAX_LEN)
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            test_correct += (logits.argmax(1) == y).sum().item()
            test_total += len(batch)
    print(f"\n  Test accuracy: {test_correct / test_total:.3f}")

    # Extract CLS embeddings for scoring
    print("\n  Extracting CLS embeddings...")
    model.eval()
    all_data = train_data + test_data
    cls_embeddings = []
    cls_labels = []
    with torch.no_grad():
        for i in range(0, len(all_data), BATCH_SIZE):
            batch = all_data[i : i + BATCH_SIZE]
            x, y = collate(batch, vocab, MAX_LEN)
            x = x.to(DEVICE)
            mask = x == 0
            pos = torch.arange(x.size(1), device=DEVICE)
            h = model.token_emb(x) + model.pos_emb(pos)
            h = model.encoder(h, src_key_padding_mask=mask)
            cls_embeddings.append(h[:, 0].cpu().numpy())
            cls_labels.extend([label for _, label in batch])
    cls_embeddings = np.concatenate(cls_embeddings, axis=0)
    cls_labels = np.array(cls_labels)
    print(f"  CLS embeddings: {cls_embeddings.shape}, labels: {cls_labels.shape}")

    # Save model + vocab + config + CLS embeddings
    config = {
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF,
        "max_len": MAX_LEN,
        "num_classes": NUM_CLASSES,
        "vocab_size": actual_vocab_size,
    }
    # Convert state_dict values to numpy for the candidate subprocess
    weights_np = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    torch.save(
        {
            "config": config,
            "vocab": vocab,
            "state_dict": model.state_dict(),
            "weights_np": weights_np,
            "cls_embeddings": cls_embeddings,
            "cls_labels": cls_labels,
        },
        SAVE_PATH,
    )
    print(f"\n  Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
