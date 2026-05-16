"""Three-way Semantic Divergence Index (SDI) computation.

SDI is the absolute distance between two agents' continuous sentiment scores.
With three agents (lexicon / expert / reasoner) we get three pairwise SDIs:

    SDI_LE = |s_vader   - s_finbert|   Lexicon vs Expert
    SDI_LR = |s_vader   - s_llm    |   Lexicon vs Reasoner
    SDI_ER = |s_finbert - s_llm    |   Expert  vs Reasoner

The four-quadrant classification then partitions samples by (SDI_LE, SDI_ER)
into committee-behaviour archetypes. See `docs/05_LAYER_L2_SPEC.md`.
"""
from __future__ import annotations

import pandas as pd

from ..config import SDI_HIGH, SDI_LOW

QUADRANT_NAMES = ('consensus', 'domain_shift', 'ambiguous', 'mixed')


def classify_quadrant(le: float, er: float, hi: float = SDI_HIGH, lo: float = SDI_LOW) -> str:
    """Map (SDI_LE, SDI_ER) → one of {'consensus', 'domain_shift', 'ambiguous', 'mixed'}.

    `ambiguous` takes priority — if Expert and Reasoner disagree strongly,
    it doesn't matter what Lexicon thinks (per L2 spec, avoid double-counting).
    """
    if er > hi:
        return 'ambiguous'
    if le < lo and er < lo:
        return 'consensus'
    if le > hi and er < lo:
        return 'domain_shift'
    return 'mixed'


def add_sdi_columns(
    df: pd.DataFrame,
    *,
    vader_col: str = 'vader_score',
    finbert_col: str = 'finbert_score',
    llm_col: str = 'llm_score',
    hi: float = SDI_HIGH,
    lo: float = SDI_LOW,
) -> pd.DataFrame:
    """Append SDI_{LE,LR,ER}, sdi_max, sdi_mean, and `quadrant` columns.

    Returns a new DataFrame (does not mutate input).
    """
    out = df.copy()
    out['sdi_le'] = (out[vader_col] - out[finbert_col]).abs()
    out['sdi_lr'] = (out[vader_col] - out[llm_col]).abs()
    out['sdi_er'] = (out[finbert_col] - out[llm_col]).abs()
    out['sdi_max'] = out[['sdi_le', 'sdi_lr', 'sdi_er']].max(axis=1)
    out['sdi_mean'] = out[['sdi_le', 'sdi_lr', 'sdi_er']].mean(axis=1)
    out['quadrant'] = [
        classify_quadrant(le, er, hi=hi, lo=lo)
        for le, er in zip(out['sdi_le'], out['sdi_er'])
    ]
    return out
