"""Mapping from sentence-level predictions to weekly trading signals.

The L5 backtest treats FPB sentences as a controlled "news proxy": each
trading week we sample N sentences from sdi_data.csv and aggregate the
chosen agent's predictions into a single BUY / SHORT / HOLD signal.

This isolates the sentiment-classification quality dimension from the
news-arrival process — see docs/08 limitations note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# How many "news sentences" we synthesise per trading week
DEFAULT_SENTENCES_PER_WEEK = 20


def aggregate_to_signal(predictions: pd.Series, threshold: float = 0.20) -> str:
    """pos/neg/neu labels → BUY/SHORT/HOLD.

    bullish_score = (#positive - #negative) / N
    > +threshold ⇒ BUY,  < -threshold ⇒ SHORT,  else HOLD
    """
    if len(predictions) == 0:
        return 'HOLD'
    n_pos = int((predictions == 'positive').sum())
    n_neg = int((predictions == 'negative').sum())
    bullish = (n_pos - n_neg) / len(predictions)
    if bullish >  threshold: return 'BUY'
    if bullish < -threshold: return 'SHORT'
    return 'HOLD'


def build_weekly_signals(
    sentences: pd.DataFrame,
    label_col: str,
    weekly_dates: pd.DatetimeIndex,
    *,
    n_per_week: int = DEFAULT_SENTENCES_PER_WEEK,
    threshold: float = 0.20,
    seed: int = 42,
) -> list[tuple[pd.Timestamp, str]]:
    """Sample N sentences per week, aggregate via `label_col` → signal.

    Args:
        sentences:    DataFrame with at least `label_col` column.
        label_col:    which prediction column to use ('vader_label',
                      'finbert_label', 'llm_label', 'critic_label', etc.).
        weekly_dates: end-of-week dates for which to emit signals.
        n_per_week:   how many sentences to synthesize per week.

    Returns:
        list of (date, signal) tuples, one per week.
    """
    if label_col not in sentences.columns:
        raise KeyError(f"{label_col!r} not in sentences.columns ({list(sentences.columns)[:6]}...)")
    rng = np.random.default_rng(seed)
    n_total = len(sentences)
    signals = []
    for d in weekly_dates:
        idx = rng.integers(0, n_total, size=n_per_week)
        preds = sentences.iloc[idx][label_col]
        signals.append((d, aggregate_to_signal(preds, threshold=threshold)))
    return signals
