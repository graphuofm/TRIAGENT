"""N-gram trigger mining via log-odds + z-score, supporting unigrams,
bigrams, and trigrams.

The 'high-SDI' subset is treated as the positive class; 'low-SDI' as the
reference. Triggers with the highest signed z-score are the n-grams whose
presence most predicts that the committee will disagree (= the case the
edge predictor in L4 must catch).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ..config import LEXICONS_DIR


def _log_odds_z(
    counts_pos: np.ndarray, counts_neg: np.ndarray,
    n_pos: int, n_neg: int, alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed log-odds and z-score of feature presence in pos vs neg."""
    p_pos = (counts_pos + alpha) / (n_pos + 2 * alpha)
    p_neg = (counts_neg + alpha) / (n_neg + 2 * alpha)
    log_odds = np.log(p_pos / (1 - p_pos)) - np.log(p_neg / (1 - p_neg))
    var = 1.0 / (counts_pos + alpha) + 1.0 / (counts_neg + alpha)
    z = log_odds / np.sqrt(var)
    return log_odds, z


def mine_ngram_triggers(
    df: pd.DataFrame,
    *,
    target_col: str = 'sdi_max',
    target_threshold: float = 0.7,
    text_col: str = 'sentence',
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 3,
    max_features: int = 25_000,
) -> pd.DataFrame:
    """Returns triggers ranked by absolute z-score, columns:
        ngram, n, presence_high_sdi, presence_low_sdi,
        log_odds, z_score, prevalence
    """
    is_pos = (df[target_col] > target_threshold).to_numpy()
    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Empty class: pos={n_pos}, neg={n_neg}")

    cv = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        token_pattern=r'[a-zA-Z]{3,}',
        lowercase=True,
    )
    X = cv.fit_transform(df[text_col].fillna('').astype(str))
    Xb = (X > 0).astype(np.int64)  # binary presence
    counts_pos = np.asarray(Xb[is_pos].sum(axis=0)).ravel()
    counts_neg = np.asarray(Xb[~is_pos].sum(axis=0)).ravel()
    log_odds, z = _log_odds_z(counts_pos, counts_neg, n_pos, n_neg)

    feats = cv.get_feature_names_out()
    ns = np.array([term.count(' ') + 1 for term in feats])
    out = pd.DataFrame({
        'ngram': feats,
        'n': ns,
        'presence_high_sdi': counts_pos,
        'presence_low_sdi':  counts_neg,
        'log_odds': log_odds,
        'z_score': z,
        'prevalence': (counts_pos + counts_neg) / (n_pos + n_neg),
    })
    return out.sort_values('z_score', ascending=False).reset_index(drop=True)


def save_top_triggers(
    triggers: pd.DataFrame,
    *,
    out_path: Path | None = None,
    top_per_n: int = 50,
) -> Path:
    """Save top-N triggers per n-gram order as JSON for L4 features."""
    out_path = out_path or LEXICONS_DIR / 'trigger_ngrams.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for n in sorted(triggers['n'].unique()):
        sub = triggers[triggers['n'] == n]
        # Top-N by absolute z-score (both directions of effect)
        top_pos = sub.nlargest(top_per_n, 'z_score')
        top_neg = sub.nsmallest(top_per_n, 'z_score')
        payload[f'n{n}_high_sdi'] = top_pos[['ngram', 'z_score', 'log_odds',
                                             'presence_high_sdi', 'presence_low_sdi']].to_dict('records')
        payload[f'n{n}_low_sdi']  = top_neg[['ngram', 'z_score', 'log_odds',
                                             'presence_high_sdi', 'presence_low_sdi']].to_dict('records')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return out_path
