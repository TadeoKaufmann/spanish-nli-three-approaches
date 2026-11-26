"""Download XNLI (Spanish) from the Hugging Face Hub and rebuild the exact 300-instance test sample.

    python scripts/get_data.py

Writes data/train.jsonl (392,702 pairs), data/eval_full.jsonl (7,500 = test + validation) and data/test_300.jsonl,
then checks the sample against results/test_300_ids.csv.

`idx` numbers the 7,500 evaluation pairs: 0..5009 are the XNLI test split, 5010..7499 the validation split.
"""
import argparse
import csv
import random
from collections import defaultdict

from common import DATA, LABELS, RESULTS, load_jsonl, write_jsonl

SEED, PER_CLASS = 42, 100


def to_rows(split, offset=0):
    return [{"idx": offset + i, "premise": r["premise"], "hypothesis": r["hypothesis"],
             "label": LABELS[r["label"]], "label_int": r["label"]} for i, r in enumerate(split)]


def stratified_sample(rows, per_class=PER_CLASS, seed=SEED):
    """Same number of pairs per class, drawn with one seeded generator, then shuffled."""
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label_int"]].append(row)
    rng = random.Random(seed)
    sample = []
    for _, group in sorted(by_label.items()):
        rng.shuffle(group)
        sample.extend(group[:per_class])
    rng.shuffle(sample)
    return sample


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from datasets import load_dataset
    xnli = load_dataset("facebook/xnli", "es")

    train = to_rows(xnli["train"])
    full = to_rows(xnli["test"]) + to_rows(xnli["validation"], offset=len(xnli["test"]))
    sample = stratified_sample(full)
    write_jsonl(DATA / "train.jsonl", train)
    write_jsonl(DATA / "eval_full.jsonl", full)
    write_jsonl(DATA / "test_300.jsonl", sample)
    print(f"train {len(train):,} | evaluation {len(full):,} | sample {len(sample)}")

    ids_file = RESULTS / "test_300_ids.csv"
    with open(ids_file, encoding="utf-8") as f:
        expected = [(int(r["idx"]), r["label"]) for r in csv.DictReader(f)]
    assert [(r["idx"], r["label"]) for r in load_jsonl(DATA / "test_300.jsonl")] == expected, (
        f"the rebuilt sample differs from {ids_file}")
    print("sample matches results/test_300_ids.csv (same pairs, same order)")


if __name__ == "__main__":
    main()
