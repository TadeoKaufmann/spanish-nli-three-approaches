"""LLM approach: prepare label-free batches for a chat model and merge its answers. Nothing here calls an API.

    python scripts/llm_batches.py split                      # data/llm_batches/batch_00.jsonl ... batch_09.jsonl
    python scripts/llm_batches.py merge --name llm_rerun     # reads preds_batch_XX.jsonl next to them

Give each batch to the model in a separate, fresh context, with no access to the rest of the repository (the labels
are in it). Ask for one JSON object per line, {"idx": ..., "pred_label": "entailment" | "neutral" | "contradiction"},
and save the answer as preds_batch_XX.jsonl.
"""
import argparse

from common import DATA, LABELS, PREDICTIONS, load_jsonl, write_jsonl

BATCH_SIZE = 30
BATCH_DIR = DATA / "llm_batches"


def make_batches(rows, size=BATCH_SIZE):
    """Pairs without their labels, in the sample's order."""
    return [[{k: r[k] for k in ("idx", "premise", "hypothesis")} for r in rows[i:i + size]]
            for i in range(0, len(rows), size)]


def merge_answers(sample, answers):
    """Check that the answers cover the sample exactly once with valid labels, and attach the gold labels."""
    by_idx = {}
    for row in answers:
        assert row["idx"] not in by_idx, f"idx {row['idx']} answered twice"
        assert row["pred_label"] in LABELS, f"idx {row['idx']}: unknown label {row['pred_label']!r}"
        by_idx[row["idx"]] = row["pred_label"]
    expected = {r["idx"] for r in sample}
    assert set(by_idx) == expected, (f"{len(expected - set(by_idx))} pairs missing, "
                                     f"{len(set(by_idx) - expected)} unknown ids")
    return [{"idx": r["idx"], "true_label": r["label"], "pred_label": by_idx[r["idx"]]} for r in sample]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["split", "merge"])
    ap.add_argument("--name", default="llm_rerun", help="prediction file name for `merge`")
    a = ap.parse_args()
    sample = load_jsonl(DATA / "test_300.jsonl")
    if a.command == "split":
        for i, batch in enumerate(make_batches(sample)):
            write_jsonl(BATCH_DIR / f"batch_{i:02d}.jsonl", batch)
        print(f"wrote {len(make_batches(sample))} batches to {BATCH_DIR}")
    else:
        answers = [row for path in sorted(BATCH_DIR.glob("preds_batch_*.jsonl")) for row in load_jsonl(path)]
        rows = merge_answers(sample, answers)
        write_jsonl(PREDICTIONS / f"{a.name}.jsonl", rows)
        correct = sum(r["true_label"] == r["pred_label"] for r in rows)
        print(f"{correct}/{len(rows)} correct -> {PREDICTIONS / (a.name + '.jsonl')}")


if __name__ == "__main__":
    main()
