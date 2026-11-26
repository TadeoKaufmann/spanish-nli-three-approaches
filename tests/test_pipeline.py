"""Fast tests that need neither the dataset nor a GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import analyze  # noqa: E402
from classic import handcrafted_features  # noqa: E402
from get_data import stratified_sample  # noqa: E402
from llm_batches import make_batches, merge_answers  # noqa: E402


def test_handcrafted_features_on_a_known_pair():
    overlap, p_neg, h_neg, diff, ratio, bigram = handcrafted_features(
        "El gato no duerme en la cama.", "El gato duerme.")
    assert (overlap, p_neg, h_neg, diff) == (1.0, 1.0, 0.0, 4.0)
    assert ratio == pytest.approx(3 / 7)
    assert bigram == pytest.approx(0.5)             # "el gato" appears in the premise, "gato duerme" does not


def test_handcrafted_features_on_empty_text():
    assert handcrafted_features("", "") == [0.0] * 6


def test_sample_is_balanced_and_deterministic():
    rows = [{"idx": i, "label_int": i % 3} for i in range(90)]
    a = stratified_sample(list(rows), per_class=10)
    assert a == stratified_sample(list(rows), per_class=10)
    assert [sum(r["label_int"] == k for r in a) for k in range(3)] == [10, 10, 10]


def test_batches_carry_no_labels():
    sample = [{"idx": i, "premise": "p", "hypothesis": "h", "label": "neutral"} for i in range(65)]
    batches = make_batches(sample)
    assert [len(b) for b in batches] == [30, 30, 5]
    assert all(set(r) == {"idx", "premise", "hypothesis"} for b in batches for r in b)


def test_merge_rejects_incomplete_or_invalid_answers():
    sample = [{"idx": 1, "label": "neutral"}, {"idx": 2, "label": "entailment"}]
    good = [{"idx": 2, "pred_label": "neutral"}, {"idx": 1, "pred_label": "neutral"}]
    assert [r["pred_label"] for r in merge_answers(sample, good)] == ["neutral", "neutral"]
    for bad in (good[:1], good + good[:1], [good[0], {"idx": 1, "pred_label": "maybe"}]):
        with pytest.raises(AssertionError):
            merge_answers(sample, bad)


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    true, pred = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    assert analyze.macro_f1(analyze.confusion(true, pred)) == pytest.approx(f1_score(true, pred, average="macro"))


def test_mcnemar_counts_only_disagreements():
    true = np.zeros(20, dtype=int)
    a = np.array([0] * 8 + [1] * 12)
    b = np.zeros(20, dtype=int)
    only_a, only_b, p = analyze.mcnemar_exact(true, a, b)
    assert (only_a, only_b) == (0, 12)
    assert p == pytest.approx(2 * 0.5 ** 12)
    assert analyze.mcnemar_exact(true, a, a) == (0, 0, 1.0)


def test_majority_vote_uses_fallback_on_three_way_ties():
    preds = [np.array([0, 0]), np.array([0, 1]), np.array([1, 2])]
    assert list(analyze.majority_vote(preds, fallback=preds[2])) == [0, 2]


def test_committed_predictions_give_the_reported_accuracy():
    order, true, preds = analyze.load_predictions()
    assert len(order) == 300
    assert {n: int((p == true).sum()) for n, p in preds.items()} == {
        "classic_overlap": 146, "beto": 241, "llm": 256}
