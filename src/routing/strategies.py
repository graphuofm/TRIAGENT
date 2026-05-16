"""Routing strategies S0-S6 over committee data.

Each strategy reads pre-computed agent outputs from `sdi_data.csv` and
returns per-row predictions plus aggregated cost / latency, so a sweep
can simulate "what if we deployed strategy X on this corpus" with no
GPU calls.

Cost model:
- Each agent always pays its own per-sample cost when invoked.
- Routing strategies that escalate compose costs additively (an L2 call
  on a sample that escalated from L1 costs vader_cost + finbert_cost).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class RouteResult:
    name: str
    predictions: np.ndarray            # final string label per row
    per_sample_cost: np.ndarray        # USD per row
    per_sample_latency: np.ndarray     # ms per row
    coverage: dict = field(default_factory=dict)  # which tier handled what fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_columns(suffix: str | None) -> tuple[str, str, str, str]:
    """(label, score, conf, cost) column names for a given LLM suffix.
    suffix=None means the canonical 7B columns ('llm_*'), else 'llm_<suffix>_*'.
    """
    if suffix is None:
        return ('llm_label', 'llm_score', 'llm_confidence', 'llm_cost_usd')
    return (
        f'llm_{suffix}_label',
        f'llm_{suffix}_score',
        f'llm_{suffix}_confidence',
        f'llm_{suffix}_cost_usd',
    )


def _llm_latency_col(suffix: str | None) -> str:
    return 'llm_latency_ms' if suffix is None else f'llm_{suffix}_latency_ms'


# ---------------------------------------------------------------------------
# S0-S2: degenerate single-agent baselines
# ---------------------------------------------------------------------------

def route_always_l1(df: pd.DataFrame, **_) -> RouteResult:
    n = len(df)
    return RouteResult(
        name='S0:always_L1',
        predictions=df['vader_label'].to_numpy(),
        per_sample_cost=df['vader_cost_usd'].to_numpy(),
        per_sample_latency=df['vader_latency_ms'].to_numpy(),
        coverage={'l1': 1.0, 'l2': 0.0, 'l3': 0.0},
    )


def route_always_l2(df: pd.DataFrame, **_) -> RouteResult:
    return RouteResult(
        name='S1:always_L2',
        predictions=df['finbert_label'].to_numpy(),
        per_sample_cost=df['finbert_cost_usd'].to_numpy(),
        per_sample_latency=df['finbert_latency_ms'].to_numpy(),
        coverage={'l1': 0.0, 'l2': 1.0, 'l3': 0.0},
    )


def route_always_l3(df: pd.DataFrame, llm_suffix: str | None = None, **_) -> RouteResult:
    label_col, _, _, cost_col = _llm_columns(llm_suffix)
    return RouteResult(
        name=f'S2:always_L3_{llm_suffix or "7b"}',
        predictions=df[label_col].to_numpy(),
        per_sample_cost=df[cost_col].to_numpy(),
        per_sample_latency=df[_llm_latency_col(llm_suffix)].to_numpy(),
        coverage={'l1': 0.0, 'l2': 0.0, 'l3': 1.0},
    )


# ---------------------------------------------------------------------------
# S3-S5: single-stage L1->L2 escalation strategies
# ---------------------------------------------------------------------------

def _escalate_l1_to_l2(df: pd.DataFrame, escalate_mask: np.ndarray, name: str) -> RouteResult:
    """Helper: build a S3/S4/S5 result given a boolean mask of which rows escalate L1->L2."""
    n = len(df)
    preds = np.where(escalate_mask, df['finbert_label'].to_numpy(),
                                     df['vader_label'].to_numpy())
    # Cost: VADER always pays (we still ran it as the gate);
    #       FinBERT pays only on escalated rows.
    cost = df['vader_cost_usd'].to_numpy() + escalate_mask * df['finbert_cost_usd'].to_numpy()
    lat = df['vader_latency_ms'].to_numpy() + escalate_mask * df['finbert_latency_ms'].to_numpy()
    return RouteResult(
        name=name,
        predictions=preds,
        per_sample_cost=cost,
        per_sample_latency=lat,
        coverage={'l1': 1 - escalate_mask.mean(),
                  'l2': escalate_mask.mean(),
                  'l3': 0.0},
    )


def route_random_pct(df: pd.DataFrame, escalation_pct: float, seed: int = 42, **_) -> RouteResult:
    """S3: randomly escalate `escalation_pct` of rows from L1 to L2."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < escalation_pct
    return _escalate_l1_to_l2(df, mask, name=f'S3:random_{int(escalation_pct*100)}%')


def route_confidence_pct(df: pd.DataFrame, escalation_pct: float, **_) -> RouteResult:
    """S4: escalate the lowest-confidence VADER samples."""
    n = len(df)
    k = int(round(n * escalation_pct))
    if k == 0:
        return _escalate_l1_to_l2(df, np.zeros(n, dtype=bool),
                                  name=f'S4:conf_{int(escalation_pct*100)}%')
    if k >= n:
        return _escalate_l1_to_l2(df, np.ones(n, dtype=bool),
                                  name=f'S4:conf_{int(escalation_pct*100)}%')
    threshold = np.partition(df['vader_confidence'].to_numpy(), k)[k]
    mask = df['vader_confidence'].to_numpy() < threshold
    # Tie-breaking: if too few escalated due to ties, pad to exact k via stable order
    while mask.sum() < k:
        idx = np.flatnonzero(~mask & (df['vader_confidence'].to_numpy() == threshold))
        if len(idx) == 0:
            break
        mask[idx[: k - mask.sum()]] = True
    return _escalate_l1_to_l2(df, mask, name=f'S4:conf_{int(escalation_pct*100)}%')


def route_sdi_pct(df: pd.DataFrame, escalation_pct: float, sdi_col: str = 'sdi_le', **_) -> RouteResult:
    """S5: escalate the highest-SDI samples (default SDI_LE)."""
    n = len(df)
    k = int(round(n * escalation_pct))
    sdi_vals = df[sdi_col].to_numpy()
    if k == 0:
        mask = np.zeros(n, dtype=bool)
    elif k >= n:
        mask = np.ones(n, dtype=bool)
    else:
        thresh = np.partition(-sdi_vals, k)[k]  # k-th largest = -k-th smallest of -x
        mask = -sdi_vals < thresh  # i.e. sdi_vals > -thresh
        # Fix ties to exact k via stable index
        if mask.sum() != k:
            order = np.argsort(-sdi_vals, kind='stable')
            mask = np.zeros(n, dtype=bool)
            mask[order[:k]] = True
    return _escalate_l1_to_l2(df, mask, name=f'S5:sdi_{sdi_col}_{int(escalation_pct*100)}%')


# ---------------------------------------------------------------------------
# S6: two-stage SDI routing  L1 -> L2 -> L3
# ---------------------------------------------------------------------------

def route_two_stage(
    df: pd.DataFrame,
    pct_l1_to_l2: float,
    pct_l2_to_l3: float,
    *,
    sdi_le_col: str = 'sdi_le',
    sdi_er_col: str = 'sdi_er',
    llm_suffix: str | None = None,
) -> RouteResult:
    """S6: stage 1 escalates top-SDI_LE pct from L1 to L2;
            stage 2 escalates top-SDI_ER pct *within the escalated set* from L2 to L3.

    `pct_l2_to_l3` is interpreted as a fraction of the L2 set, not the full corpus.
    """
    n = len(df)
    label_col, _, _, cost_col = _llm_columns(llm_suffix)

    sdi_le = df[sdi_le_col].to_numpy()
    sdi_er = df[sdi_er_col].to_numpy()

    # Stage 1: pick top pct_l1_to_l2 by SDI_LE
    k1 = int(round(n * pct_l1_to_l2))
    if k1 == 0:
        l2_mask = np.zeros(n, dtype=bool)
    elif k1 >= n:
        l2_mask = np.ones(n, dtype=bool)
    else:
        order = np.argsort(-sdi_le, kind='stable')
        l2_mask = np.zeros(n, dtype=bool)
        l2_mask[order[:k1]] = True

    # Stage 2: among the L2 set, pick top pct_l2_to_l3 by SDI_ER
    l3_mask = np.zeros(n, dtype=bool)
    l2_indices = np.flatnonzero(l2_mask)
    k2 = int(round(len(l2_indices) * pct_l2_to_l3))
    if k2 > 0 and len(l2_indices) > 0:
        sub_sdi_er = sdi_er[l2_indices]
        order2 = np.argsort(-sub_sdi_er, kind='stable')
        l3_mask[l2_indices[order2[:k2]]] = True
    # An L3 sample is also implicitly in L2; but we replace its label with the LLM's
    # and pay the LLM cost on top.

    # Predictions: start with VADER, override with FinBERT on l2_mask, override with LLM on l3_mask
    preds = df['vader_label'].to_numpy().copy()
    preds[l2_mask] = df['finbert_label'].to_numpy()[l2_mask]
    preds[l3_mask] = df[label_col].to_numpy()[l3_mask]

    # Cost: VADER always; FinBERT on l2_mask; LLM on l3_mask (additive)
    cost = (
        df['vader_cost_usd'].to_numpy()
        + l2_mask * df['finbert_cost_usd'].to_numpy()
        + l3_mask * df[cost_col].to_numpy()
    )
    lat = (
        df['vader_latency_ms'].to_numpy()
        + l2_mask * df['finbert_latency_ms'].to_numpy()
        + l3_mask * df[_llm_latency_col(llm_suffix)].to_numpy()
    )

    return RouteResult(
        name=f'S6:two_stage_{int(pct_l1_to_l2*100)}%/{int(pct_l2_to_l3*100)}%',
        predictions=preds,
        per_sample_cost=cost,
        per_sample_latency=lat,
        coverage={
            'l1': float(1 - l2_mask.mean()),
            'l2': float(l2_mask.mean() - l3_mask.mean()),
            'l3': float(l3_mask.mean()),
        },
    )


# ---------------------------------------------------------------------------
# Evaluation utility
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, result: RouteResult) -> dict:
    gold = df['label_text'].to_numpy()
    return {
        'name': result.name,
        'acc': accuracy_score(gold, result.predictions),
        'f1_macro': f1_score(gold, result.predictions, average='macro'),
        'cost_per_1k': float(result.per_sample_cost.sum()) / len(df) * 1000,
        'mean_latency_ms': float(result.per_sample_latency.mean()),
        'l1_pct': result.coverage.get('l1', 0.0),
        'l2_pct': result.coverage.get('l2', 0.0),
        'l3_pct': result.coverage.get('l3', 0.0),
    }
