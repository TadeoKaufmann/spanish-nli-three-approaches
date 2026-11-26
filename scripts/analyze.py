"""Compare the approaches on the 300-pair sample from the saved predictions (no data download needed for the tables).

    python scripts/analyze.py

Writes to results/: metrics.csv (with bootstrap intervals), pairwise_tests.csv (exact McNemar), agreement.csv,
llm_batches.csv and, if data/test_300.jsonl exists, slices.csv. Figures go to figures/.
"""
import csv
import itertools
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from common import DATA, FIGURES, LABELS, PREDICTIONS, RESULTS, load_jsonl

APPROACHES = {"classic_overlap": "Classic (TF-IDF + features)", "beto": "BETO fine-tuned",
              "llm": "LLM zero-shot"}
SHORT = {"classic_overlap": "Classic", "beto": "BETO", "llm": "LLM"}
COLORS = {"classic_overlap": "#257de0", "beto": "#f47230", "llm": "#00b894"}
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
N_BOOT, SEED = 10_000, 0


def confusion(true, pred):
    m = np.zeros((3, 3), dtype=int)
    np.add.at(m, (true, pred), 1)
    return m


def macro_f1(conf):
    """conf (..., 3, 3) -> macro F1."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    denom = conf.sum(axis=-1) + conf.sum(axis=-2)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(axis=-1)


def accuracy(true, pred):
    return float((true == pred).mean())


def bootstrap(true, preds, n_boot=N_BOOT, seed=SEED):
    """Resample the pairs once and reuse the draws for every approach, so differences are paired."""
    draws = np.random.default_rng(seed).integers(0, len(true), size=(n_boot, len(true)))
    out = {}
    for name, pred in preds.items():
        cell = (true * 3 + pred)[draws]                                 # (n_boot, n) values in 0..8
        conf = np.stack([(cell == c).sum(axis=1) for c in range(9)], axis=1).reshape(n_boot, 3, 3)
        out[name] = {"accuracy": (true == pred)[draws].mean(axis=1), "f1_macro": macro_f1(conf)}
    return out


def mcnemar_exact(true, a, b):
    """(pairs only a gets right, pairs only b gets right, two-sided exact p-value)."""
    only_a = int(((a == true) & (b != true)).sum())
    only_b = int(((b == true) & (a != true)).sum())
    total = only_a + only_b
    return only_a, only_b, float(binomtest(only_a, total, 0.5).pvalue) if total else 1.0


def majority_vote(preds, fallback):
    """Most voted class per pair; three-way ties go to `fallback`."""
    stack = np.stack(preds)
    counts = np.stack([(stack == k).sum(axis=0) for k in range(3)])
    return np.where(counts.max(axis=0) >= 2, counts.argmax(axis=0), fallback)


def write_csv(name, header, rows):
    with open(RESULTS / name, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"\n== {name}")
    for row in [header] + rows:
        print("  ".join(str(x) for x in row))


def load_predictions():
    order, true, preds = None, None, {}
    for name in APPROACHES:
        rows = {r["idx"]: r for r in load_jsonl(PREDICTIONS / f"{name}.jsonl")}
        if order is None:
            with open(RESULTS / "test_300_ids.csv", encoding="utf-8") as f:
                ids = list(csv.DictReader(f))
            order = [int(r["idx"]) for r in ids]
            true = np.array([LABELS.index(r["label"]) for r in ids])
        assert set(rows) == set(order), f"{name} does not cover the same 300 pairs"
        assert all(rows[i]["true_label"] == LABELS[t] for i, t in zip(order, true)), f"{name}: gold labels differ"
        preds[name] = np.array([LABELS.index(rows[i]["pred_label"]) for i in order])
    return order, true, preds


def figure_scores(true, preds, samples):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = list(preds)
    for y, name in enumerate(names):
        value = accuracy(true, preds[name])
        low, high = np.percentile(samples[name]["accuracy"], [2.5, 97.5])
        ax.barh(y, value, height=0.55, color=COLORS[name])
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], color=TEXT, capsize=4, linewidth=1.2)
        ax.text(high + 0.015, y, f"{value:.3f}", va="center", fontsize=10, color=TEXT)
    ax.axvline(1 / 3, color=MUTED, linestyle="--", linewidth=1)
    ax.text(1 / 3 + 0.01, 2.5, "chance", color=MUTED, fontsize=8.5, va="center")
    ax.set_yticks(range(len(names)), [APPROACHES[n] for n in names])
    ax.set_ylim(2.65, -0.45)
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_title("Accuracy on the same 300 pairs, with 95 % bootstrap intervals", loc="left", fontsize=11.5, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy.png", dpi=150)
    plt.close(fig)


def figure_confusions(true, preds):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), layout="constrained")
    for ax, name in zip(axes, preds):
        m = confusion(true, preds[name])
        share = m / m.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i, j in itertools.product(range(3), range(3)):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", color="white" if share[i, j] > 0.55 else TEXT)
        ax.set_xticks(range(3), ["entail.", "neutral", "contra."])
        ax.set_yticks(range(3), ["entail.", "neutral", "contra."] if ax is axes[0] else ["", "", ""])
        ax.set_title(APPROACHES[name], fontsize=11, color=TEXT)
        ax.set_xlabel("predicted", color=MUTED)
        ax.tick_params(length=0, colors=MUTED)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("true", color=MUTED)
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def slices(order, true, preds):
    """Accuracy by simple surface properties of the pair. Needs the downloaded sentences."""
    from classic import NEGATION_WORDS, tokenize
    text = {r["idx"]: r for r in load_jsonl(DATA / "test_300.jsonl")}
    negated, overlap = [], []
    for i in order:
        p, h = set(tokenize(text[i]["premise"])), tokenize(text[i]["hypothesis"])
        negated.append(any(t in NEGATION_WORDS for t in h))
        overlap.append(len(set(h) & p) / len(set(h)) if h else 0.0)
    negated, overlap = np.array(negated), np.array(overlap)
    low, high = np.quantile(overlap, [1 / 3, 2 / 3])
    groups = {"hypothesis has a negation word": negated, "no negation word": ~negated,
              "word overlap: low third": overlap <= low, "word overlap: middle third": (overlap > low) & (overlap <= high),
              "word overlap: high third": overlap > high}
    rows = []
    for label, mask in groups.items():
        shares = [int((true[mask] == k).sum()) for k in range(3)]
        rows.append([label, int(mask.sum()), *shares] + [round(accuracy(true[mask], preds[n][mask]), 3) for n in preds])
    write_csv("slices.csv", ["slice", "n", "n_entailment", "n_neutral", "n_contradiction"] + [SHORT[n] for n in preds], rows)


def main():
    FIGURES.mkdir(exist_ok=True)
    order, true, preds = load_predictions()
    samples = bootstrap(true, preds)

    rows = []
    for name, pred in preds.items():
        row = [SHORT[name]]
        for metric, value in (("accuracy", accuracy(true, pred)), ("f1_macro", macro_f1(confusion(true, pred)))):
            low, high = np.percentile(samples[name][metric], [2.5, 97.5])
            row += [round(float(value), 4), round(float(low), 4), round(float(high), 4)]
        conf = confusion(true, pred)
        row += [round(float(x), 4) for x in 2 * np.diag(conf) / (conf.sum(axis=0) + conf.sum(axis=1))]
        rows.append(row)
    write_csv("metrics.csv", ["approach", "accuracy", "acc_low", "acc_high", "f1_macro", "f1_low", "f1_high"]
              + [f"f1_{label}" for label in LABELS], rows)

    rows = []
    for a, b in itertools.combinations(preds, 2):
        only_a, only_b, p = mcnemar_exact(true, preds[a], preds[b])
        diff = samples[b]["accuracy"] - samples[a]["accuracy"]
        low, high = np.percentile(diff, [2.5, 97.5])
        rows.append([SHORT[a], SHORT[b], round(accuracy(true, preds[b]) - accuracy(true, preds[a]), 4),
                     round(float(low), 4), round(float(high), 4), only_a, only_b, float(f"{p:.3g}")])
    write_csv("pairwise_tests.csv", ["a", "b", "accuracy_b_minus_a", "diff_low", "diff_high", "only_a_right",
                                     "only_b_right", "mcnemar_p"], rows)

    correct = np.stack([preds[n] == true for n in preds])
    vote = majority_vote(list(preds.values()), preds["llm"])
    oracle = correct.any(axis=0)
    rows = [["all three right", int(correct.all(axis=0).sum())], ["all three wrong", int((~oracle).sum())],
            ["at least one right", int(oracle.sum())],
            ["BETO and LLM agree", int((preds["beto"] == preds["llm"]).sum())],
            ["... and are right", int(((preds["beto"] == preds["llm"]) & (preds["llm"] == true)).sum())],
            ["majority vote right (ties go to the LLM)", int((vote == true).sum())]]
    write_csv("agreement.csv", ["count_out_of_300", "n"], rows)

    batch = np.arange(len(order)) // 30                                  # the LLM saw the sample in order, 30 at a time
    rows = [[int(b), round(accuracy(true[batch == b], preds["llm"][batch == b]), 3),
             round(accuracy(true[batch == b], preds["beto"][batch == b]), 3)] for b in np.unique(batch)]
    write_csv("llm_batches.csv", ["batch", "llm_accuracy", "beto_accuracy_same_pairs"], rows)

    figure_scores(true, preds, samples)
    figure_confusions(true, preds)
    if (DATA / "test_300.jsonl").exists():
        slices(order, true, preds)
    else:
        print("\nslices.csv skipped: run scripts/get_data.py first")


if __name__ == "__main__":
    main()
