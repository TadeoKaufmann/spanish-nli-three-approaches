"""Classical approach: logistic regression on shallow features, plus two probes added later.

    python scripts/classic.py                     # all three variants

Variants
  overlap      (original) TF-IDF cosine between premise and hypothesis + six handcrafted features: 7 numbers.
  hypothesis   TF-IDF of the hypothesis alone. The premise is never read, so anything above chance comes from
               cues in how hypotheses were written.
  combined     the 7 features of `overlap` next to the hypothesis TF-IDF.

Writes predictions for the 300-instance sample and for all 7,500 evaluation pairs, and results/classic_runs.json.
"""
import argparse
import json
import re
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from common import DATA, PREDICTIONS, RESULTS, load_jsonl, prediction_rows, write_jsonl

NEGATION_WORDS = {"no", "nunca", "jamás", "jamas", "nada", "tampoco", "ni", "sin"}
VARIANTS = ["overlap", "hypothesis", "combined"]


def tokenize(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def bigrams(tokens):
    return set(zip(tokens, tokens[1:]))


def handcrafted_features(premise, hypothesis):
    """[word overlap, premise negations, hypothesis negations, length difference, length ratio, bigram overlap]"""
    p_tokens, h_tokens = tokenize(premise), tokenize(hypothesis)
    p_set, h_set = set(p_tokens), set(h_tokens)
    overlap = len(h_set & p_set) / len(h_set) if h_set else 0.0
    p_neg = sum(t in NEGATION_WORDS for t in p_tokens)
    h_neg = sum(t in NEGATION_WORDS for t in h_tokens)
    length_diff = len(p_tokens) - len(h_tokens)
    length_ratio = len(h_tokens) / len(p_tokens) if p_tokens else 0.0
    h_bigrams = bigrams(h_tokens)
    bigram_overlap = len(h_bigrams & bigrams(p_tokens)) / len(h_bigrams) if h_bigrams else 0.0
    return [overlap, float(p_neg), float(h_neg), float(length_diff), length_ratio, bigram_overlap]


def paired_cosine(a, b):
    """Row-wise cosine similarity of two L2-normalized TF-IDF matrices."""
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def overlap_features(rows, vectorizer):
    premises = [r["premise"] for r in rows]
    hypotheses = [r["hypothesis"] for r in rows]
    cosine = paired_cosine(vectorizer.transform(premises), vectorizer.transform(hypotheses))
    handmade = np.array([handcrafted_features(p, h) for p, h in zip(premises, hypotheses)])
    return np.hstack([cosine[:, None], handmade])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    a = ap.parse_args()

    train = load_jsonl(DATA / "train.jsonl")
    sample, full = load_jsonl(DATA / "test_300.jsonl"), load_jsonl(DATA / "eval_full.jsonl")
    y_train = np.array([r["label_int"] for r in train])

    runs_file = RESULTS / "classic_runs.json"
    runs = json.loads(runs_file.read_text(encoding="utf-8")) if runs_file.exists() else {}
    for variant in a.variants:
        t0 = time.time()
        matrices = {}
        if variant != "hypothesis":
            pair_vectorizer = TfidfVectorizer(max_features=30000)
            pair_vectorizer.fit([r["premise"] for r in train] + [r["hypothesis"] for r in train])
            dense = [overlap_features(rows, pair_vectorizer) for rows in (train, sample, full)]
            matrices["overlap"] = dense
        if variant != "overlap":
            hyp_vectorizer = TfidfVectorizer(max_features=30000)
            hyp_vectorizer.fit([r["hypothesis"] for r in train])
            matrices["hypothesis"] = [hyp_vectorizer.transform([r["hypothesis"] for r in rows])
                                      for rows in (train, sample, full)]
        if variant == "combined":
            # length difference is unbounded: scale by the training maximum so it does not dominate the TF-IDF values
            scale = np.maximum(np.abs(matrices["overlap"][0]).max(axis=0), 1e-9)
            X = [sparse.hstack([sparse.csr_matrix(d / scale), h], format="csr")
                 for d, h in zip(matrices["overlap"], matrices["hypothesis"])]
        else:
            X = matrices[variant]

        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[0], y_train)
        runs[variant] = {"seconds": round(time.time() - t0, 1), "n_train": len(train),
                         "iterations": int(model.n_iter_.max())}
        for name, rows, Xe in (("test_300", sample, X[1]), ("eval_full", full, X[2])):
            pred = model.predict(Xe)
            truth = [r["label_int"] for r in rows]
            runs[variant][name] = {"accuracy": round(accuracy_score(truth, pred), 4),
                                   "f1_macro": round(f1_score(truth, pred, average="macro"), 4)}
            suffix = "" if name == "test_300" else "_full"
            write_jsonl(PREDICTIONS / f"classic_{variant}{suffix}.jsonl", prediction_rows(rows, pred))
        print(variant, runs[variant], flush=True)
        runs_file.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
