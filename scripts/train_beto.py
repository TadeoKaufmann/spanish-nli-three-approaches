"""Deep learning approach: fine-tune BETO on XNLI Spanish and predict the evaluation pairs.

    python scripts/train_beto.py --fp16                 # full run, needs a GPU
    python scripts/train_beto.py --limit-train 96 --limit-eval 30 --dev-size 30 --epochs 1   # smoke test on CPU

This is a cleaned-up version of the training script, with the same configuration as the run that produced
results/predictions/beto.jsonl: 2,000 balanced pairs held out of train for early stopping, AdamW 2e-5, weight decay
0.01, 6 % linear warmup, batch 32, 128 tokens, up to 10 epochs with patience 3, seed 42. It writes to
results/predictions/beto_rerun*.jsonl and leaves the committed predictions untouched.
"""
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import DATA, PREDICTIONS, load_jsonl, prediction_rows, write_jsonl

MODEL_ID = "dccuchile/bert-base-spanish-wwm-cased"


def split_dev(rows, n_dev, seed):
    """(dev, train): dev is balanced across the three classes."""
    rng = random.Random(seed)
    by_label = {k: [r for r in rows if r["label_int"] == k] for k in (0, 1, 2)}
    dev, rest = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        take = n_dev // 3 + (1 if label < n_dev % 3 else 0)
        dev += group[:take]
        rest += group[take:]
    rng.shuffle(rest)
    return dev, rest


def loader(rows, tokenizer, max_length, batch_size, shuffle=False):
    def collate(batch):
        enc = tokenizer([r["premise"] for r in batch], [r["hypothesis"] for r in batch], truncation=True,
                        max_length=max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([r["label_int"] for r in batch])
        return enc
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    return np.concatenate([model(**{k: v.to(device) for k, v in b.items() if k != "labels"}).logits.argmax(-1).cpu().numpy()
                           for b in dl])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--dev-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-train", type=int, help="use only this many training pairs (smoke test)")
    ap.add_argument("--limit-eval", type=int, help="predict only this many evaluation pairs (smoke test)")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = a.fp16 and device.type == "cuda"

    dev, train = split_dev(load_jsonl(DATA / "train.jsonl"), a.dev_size, a.seed)
    train = train[:a.limit_train]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3).to(device)

    train_dl = loader(train, tokenizer, a.max_length, a.batch_size, shuffle=True)
    dev_dl = loader(dev, tokenizer, a.max_length, a.batch_size * 2)
    dev_labels = np.array([r["label_int"] for r in dev])
    steps = len(train_dl) * a.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.06), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"device {device} | train {len(train):,} | dev {len(dev):,} | {len(train_dl):,} steps per epoch")

    best, best_state, waited = -1.0, None, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item()
        dev_acc = float((predict(model, dev_dl, device) == dev_labels).mean())
        print(f"epoch {epoch} | loss {total / len(train_dl):.4f} | dev accuracy {dev_acc:.4f}", flush=True)
        if dev_acc > best:
            best, waited = dev_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= a.patience:
                break
    model.load_state_dict(best_state)

    smoke = a.limit_train is not None or a.limit_eval is not None
    for source, suffix in (("test_300", ""), ("eval_full", "_full")):
        rows = load_jsonl(DATA / f"{source}.jsonl")[:a.limit_eval]
        pred = predict(model, loader(rows, tokenizer, a.max_length, a.batch_size * 2), device)
        accuracy = float((pred == np.array([r["label_int"] for r in rows])).mean())
        out = PREDICTIONS / f"beto_{'smoke' if smoke else 'rerun'}{suffix}.jsonl"
        write_jsonl(out, prediction_rows(rows, pred))
        print(f"{source}: {len(rows):,} pairs | accuracy {accuracy:.4f} -> {out.name}")


if __name__ == "__main__":
    main()
