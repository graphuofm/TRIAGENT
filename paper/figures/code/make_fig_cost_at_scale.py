"""§1 Cost-at-scale: 100K → 10M users economic case.

Shows annual inference cost (USD, log scale) as a function of user count
for four deployment baselines and TriAgent SDI two-stage routing.

Per-query cost numbers (verified in our own measurements):
    VADER          : negligible
    FinBERT        : $0.0005 / 1k queries  (GPU-amortised)
    Qwen-7B local  : $0.0288 / 1k queries  (GPU-amortised)
    GPT-4o-mini    : $0.30   / 1k queries  (OpenAI public pricing 2025)
    GPT-4 (large)  : $10.00  / 1k queries  (OpenAI public pricing 2025)

TriAgent SDI two-stage Balanced operating point:
    routes ~70% to L1, 30% to L2, 5% (of L2) to L3 → ~15% LLM coverage
    total per-query cost ≈ 0.15 × LLM cost + 0.30 × FinBERT cost
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS, WONG


# Per-1000-query cost in USD (verified in our experiments where local;
# OpenAI public list prices for the API ones)
COST_PER_1K = {
    'FinBERT alone (already cheap)':       0.0005,
    'Qwen-7B self-hosted':                 0.0288,
    'GPT-4o-mini API':                     0.30,
    'GPT-4 API':                          10.00,
}
# TriAgent two-stage Balanced replaces "always-LLM" with mostly-V/F + 15% LLM
TRIAGENT_FRAC = 0.15


def main():
    apply_paper_style()
    # User counts spanning 4 orders of magnitude
    n_users   = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    queries_per_user_year = 10 * 365            # 10 queries/day × 365 days

    fig, ax = plt.subplots(figsize=(9.4, 6.0))

    palette = {
        'FinBERT alone (already cheap)': WONG['blue'],
        'Qwen-7B self-hosted':           WONG['purple'],
        'GPT-4o-mini API':               WONG['orange'],
        'GPT-4 API':                     WONG['vermillion'],
    }
    markers = {
        'FinBERT alone (already cheap)': 'P',
        'Qwen-7B self-hosted':           'X',
        'GPT-4o-mini API':               'D',
        'GPT-4 API':                     'o',
    }

    for name, ppk in COST_PER_1K.items():
        cost_year = n_users * queries_per_user_year * ppk / 1000
        ax.plot(n_users, cost_year, marker=markers[name], markersize=11,
                lw=2.6, color=palette[name],
                label=name)

    # TriAgent line — apply to GPT-4o-mini baseline (most realistic comparison)
    base_cost = COST_PER_1K['GPT-4o-mini API']
    finbert_cost = COST_PER_1K['FinBERT alone (already cheap)']
    triagent_per_k = TRIAGENT_FRAC * base_cost + 0.30 * finbert_cost
    triagent_year = n_users * queries_per_user_year * triagent_per_k / 1000
    ax.plot(n_users, triagent_year, marker='*', markersize=18,
            lw=3.4, color=WONG['green'],
            label=f'TriAgent (15% LLM via SDI two-stage, vs GPT-4o-mini)')

    # Annotate the savings at 10M users
    base_at_10m = n_users[-1] * queries_per_user_year * base_cost / 1000
    tri_at_10m  = n_users[-1] * queries_per_user_year * triagent_per_k / 1000
    saved = base_at_10m - tri_at_10m
    ax.annotate(
        f'Saves ${saved/1e6:.1f}M/yr\n at 10M users',
        xy=(n_users[-1], tri_at_10m), xytext=(3e5, 3),
        fontsize=14, fontweight='bold', color=WONG['green'],
        ha='left',
        arrowprops=dict(arrowstyle='->', color=WONG['green'], lw=2.0,
                        connectionstyle='arc3,rad=-0.3'),
    )

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('User count  (each making 10 queries/day)')
    ax.set_ylabel('Annual inference cost (USD, log scale)')
    ax.set_title('Inference cost at scale  —  TriAgent vs four deployment baselines',
                 pad=14)
    # Format axes nicely
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(['1K', '10K', '100K', '1M', '10M'])
    yticks = [1, 100, 10_000, 1_000_000, 100_000_000]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['$1', '$100', '$10K', '$1M', '$100M'])
    ax.set_ylim(0.5, 5e8)
    ax.legend(loc='upper left', fontsize=11.5)

    save_paper_figure(fig, 'fig_cost_at_scale')


if __name__ == '__main__':
    main()
