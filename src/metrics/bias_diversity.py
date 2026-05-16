"""Pairwise Cohen's kappa + per-sample disagreement entropy.

Used to quantify *bias diversity* across the committee (CFP topic 3): low
pairwise kappa = high diversity = the committee actually disagrees enough to
be useful as an ensemble.
"""
from __future__ import annotations

from collections import Counter
from math import log2

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def pairwise_kappa(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    """Square Cohen's kappa matrix over the given label columns."""
    n = len(label_cols)
    K = np.zeros((n, n), dtype=float)
    for i, ci in enumerate(label_cols):
        for j, cj in enumerate(label_cols):
            K[i, j] = 1.0 if i == j else cohen_kappa_score(df[ci], df[cj])
    return pd.DataFrame(K, index=label_cols, columns=label_cols)


def shannon_entropy(values) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


def add_disagreement_entropy(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    """Per-row Shannon entropy of the K-agent label distribution.

    Bounds:
        0  → all agents agree
        log2(K) → all agents disagree maximally (≈1.585 for K=3 over 3 classes)
    """
    out = df.copy()
    out['disagreement_entropy'] = [
        shannon_entropy(row) for row in out[label_cols].itertuples(index=False, name=None)
    ]
    return out


def error_set_overlap(df: pd.DataFrame, label_cols: list[str], gold_col: str) -> pd.DataFrame:
    """Pairwise Jaccard overlap of the per-agent error sets.

    Headline metric for the security/bias contribution: low overlap means
    different agents fail on different sentences ⇒ majority vote helps.
    """
    error_sets = {c: set(df.index[df[c] != df[gold_col]]) for c in label_cols}
    n = len(label_cols)
    J = np.zeros((n, n), dtype=float)
    for i, ci in enumerate(label_cols):
        for j, cj in enumerate(label_cols):
            si, sj = error_sets[ci], error_sets[cj]
            union = si | sj
            J[i, j] = (len(si & sj) / len(union)) if union else 1.0
    return pd.DataFrame(J, index=label_cols, columns=label_cols)
