import math

import pytest

from ragbench.eval.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from ragbench.eval.stats import bootstrap_ci, paired_bootstrap


def test_recall_and_hit():
    retrieved = ["a", "x", "b", "y"]
    assert recall_at_k(retrieved, {"a", "b"}, k=2) == 0.5
    assert recall_at_k(retrieved, {"a", "b"}, k=3) == 1.0
    assert hit_at_k(retrieved, {"y"}, k=3) == 0.0


def test_mrr():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_perfect_and_duplicates():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)
    # The duplicate "a" brings no extra gain
    assert ndcg_at_k(["a", "a"], {"a", "b"}, k=2) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_bootstrap_ci_contains_mean():
    ci = bootstrap_ci([0, 1] * 50)
    assert ci.low < ci.mean == 0.5 < ci.high


def test_paired_bootstrap_detects_clear_improvement():
    a = [0.0] * 60 + [1.0] * 40
    b = [1.0] * 100
    res = paired_bootstrap(a, b)
    assert res.diff.mean == pytest.approx(0.6)
    assert res.p_value < 0.01


def test_paired_bootstrap_no_difference():
    a = [0.0, 1.0] * 50
    assert paired_bootstrap(a, a).p_value == 1.0
