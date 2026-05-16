"""Confidence-weighted majority vote.

Cheapest possible interaction: no extra LLM calls, just aggregate the
predictions we already have. Each agent's vote is weighted by its
self-reported confidence; ties broken toward 'neutral'.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .base import InteractionResult


AGENT_LABEL_COLS = {
    'vader':   'vader_label',
    'finbert': 'finbert_label',
    'llm':     'llm_label',
}
AGENT_CONF_COLS = {
    'vader':   'vader_confidence',
    'finbert': 'finbert_confidence',
    'llm':     'llm_confidence',
}
AGENT_SCORE_COLS = {
    'vader':   'vader_score',
    'finbert': 'finbert_score',
    'llm':     'llm_score',
}


def vote_row(
    row: pd.Series,
    agents: tuple[str, ...] = ('vader', 'finbert', 'llm'),
    *,
    llm_label_col: str = 'llm_label',
    llm_conf_col: str = 'llm_confidence',
    llm_score_col: str = 'llm_score',
) -> InteractionResult:
    """Confidence-weighted majority vote across the given agents."""
    # Allow swapping the LLM column (e.g. for the multi-size sweep)
    label_cols = dict(AGENT_LABEL_COLS, llm=llm_label_col)
    conf_cols = dict(AGENT_CONF_COLS, llm=llm_conf_col)
    score_cols = dict(AGENT_SCORE_COLS, llm=llm_score_col)

    weighted: dict[str, float] = defaultdict(float)
    sum_score = 0.0
    sum_w = 0.0
    for a in agents:
        lbl = row[label_cols[a]]
        conf = max(float(row[conf_cols[a]]), 1e-3)  # floor so low-conf agents still register
        weighted[lbl] += conf
        sum_score += conf * float(row[score_cols[a]])
        sum_w += conf

    final_label = max(weighted.items(), key=lambda kv: (kv[1], kv[0] == 'neutral'))[0]
    # If all-tie, prefer neutral (financial-domain conservative default)
    top = sorted(weighted.values(), reverse=True)
    if len(top) >= 2 and abs(top[0] - top[1]) < 1e-9 and 'neutral' in weighted:
        final_label = 'neutral'

    score = sum_score / sum_w if sum_w > 0 else 0.0
    conf = top[0] / sum_w if sum_w > 0 else 0.0
    return InteractionResult(
        label=final_label,
        score=score,
        confidence=conf,
        rationale=f"votes={dict(weighted)}",
    )


def apply_vote(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run vote_row across the entire dataframe; returns a copy with new columns."""
    out = df.copy()
    results = [vote_row(row, **kwargs) for _, row in df.iterrows()]
    out['vote_label']      = [r.label for r in results]
    out['vote_score']      = [r.score for r in results]
    out['vote_confidence'] = [r.confidence for r in results]
    return out
