"""Paths, labels and file helpers shared by all scripts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"                  # downloaded XNLI files, not versioned
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
FIGURES = ROOT / "figures"

LABELS = ["entailment", "neutral", "contradiction"]      # index = XNLI label id


def load_jsonl(path):
    # utf-8-sig: some hand-assembled prediction files start with a byte order mark
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_rows(instances, predicted_ids):
    return [{"idx": r["idx"], "true_label": r["label"], "pred_label": LABELS[int(p)]}
            for r, p in zip(instances, predicted_ids)]
