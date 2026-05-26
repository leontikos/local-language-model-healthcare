"""Selective-prediction and calibration metrics.

All functions take 1-D numpy arrays:
  scores  — float, higher = more confident the model's answer is correct
  labels  — int 0/1, 1 = model's answer was correct

AUGRC / AURC / ECE follow the standard definitions used in the selective-prediction
literature (Geifman & El-Yaniv 2017; Traub et al. NeurIPS 2024 for AUGRC, arXiv:2407.01032).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

__all__ = [
    "compute_augrc",
    "compute_aurc",
    "compute_ece",
    "bootstrap_ci",
    "roc_auc_score",
    "brier_score_loss",
]


def compute_aurc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve.

    Sort by descending score; at coverage k/n, risk = error rate of the top-k.
    Lower is better. Range [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(labels)
    if n == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    y_sorted = labels[order]
    cum_correct = np.cumsum(y_sorted)
    k = np.arange(1, n + 1)
    risk = 1.0 - cum_correct / k
    coverage = k / n
    return float(np.trapezoid(risk, coverage))


def compute_augrc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the Generalized Risk-Coverage curve.

    AUGRC(s) = integral over coverage c in [0,1] of (c * risk(c)).
    Equivalent: expected risk of undetected failures across all coverage thresholds.
    Lower is better. Reference: Traub et al., NeurIPS 2024 (arXiv:2407.01032).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(labels)
    if n == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    y_sorted = labels[order]
    cum_correct = np.cumsum(y_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - cum_correct / k
    generalized_risk = coverage * risk
    return float(np.trapezoid(generalized_risk, coverage))


def compute_ece(probs: np.ndarray, labels: np.ndarray,
                n_bins: int = 15, strategy: str = "quantile") -> float:
    """Expected Calibration Error.

    strategy='quantile' → equal-mass bins (recommended; equal-width can leave
    bins with very few samples and inflate variance).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(labels)
    if n == 0:
        return float("nan")
    if strategy == "quantile":
        edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
        # Make right edges strictly inclusive at the last bin
        edges[-1] = edges[-1] + 1e-9
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def bootstrap_ci(metric_fn, scores: np.ndarray, labels: np.ndarray,
                 n_boot: int = 1000, seed: int = 42,
                 ci: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    """Percentile bootstrap CI for any metric_fn(scores, labels) → float."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(float(metric_fn(scores[idx], labels[idx])))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, list(ci))
    return float(lo), float(hi)
