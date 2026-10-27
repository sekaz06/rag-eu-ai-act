"""Bootstrap confidence intervals and paired comparisons between configurations.

Questions are the statistical unit: we resample questions with replacement.
For comparisons we resample the *same* questions for both systems (paired
bootstrap), which removes question difficulty from the variance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class CI:
    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


@dataclass
class PairedResult:
    diff: CI
    p_value: float


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> CI:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return CI(float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> PairedResult:
    """Compare system B against system A on the same questions (diff = B - A)."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Paired comparison needs scores on the same questions")
    d = b_arr - a_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    # Two-sided p-value: how often the resampled difference falls on the other side of 0
    p = 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    ci = CI(float(d.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))
    return PairedResult(diff=ci, p_value=min(1.0, p))
