"""L3: Token-Economic Pareto Frontier — the paper's main figure.

Sweeps S0-S6 routing strategies across escalation percentages, computes
per-strategy (cost, F1) curves, plots the Pareto frontier, and emits the
"three operating points" table for the paper.

Usage:
    python experiments/L3_routing_pareto.py
    python experiments/L3_routing_pareto.py --llm-size 3B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR, TABLES_DIR
from src.routing.strategies import (
    route_always_l1, route_always_l2, route_always_l3,
    route_random_pct, route_confidence_pct, route_sdi_pct,
    route_two_stage, evaluate,
)
from src.viz.style import apply_style, COLORS


SIZE_TO_SUFFIX = {
    '0.5B': '0p5b', '1.5B': '1p5b', '3B': '3b', '7B': None, '14B': '14b',
}


def sweep_pareto(df: pd.DataFrame, llm_suffix: str | None) -> pd.DataFrame:
    """Run the full strategy x escalation_pct sweep and return a long DataFrame."""
    rows = []

    # Degenerate baselines
    for fn in (route_always_l1, route_always_l2):
        r = fn(df)
        rows.append({**evaluate(df, r), 'strategy': r.name.split(':')[0],
                     'pct_l1_to_l2': 0.0, 'pct_l2_to_l3': 0.0})
    r = route_always_l3(df, llm_suffix=llm_suffix)
    rows.append({**evaluate(df, r), 'strategy': 'S2',
                 'pct_l1_to_l2': 0.0, 'pct_l2_to_l3': 0.0})

    # 1-D sweeps for S3, S4, S5
    pcts = np.linspace(0, 1, 21)  # 0%, 5%, ..., 100%
    for pct in pcts:
        for tag, fn in [('S3', route_random_pct),
                        ('S4', route_confidence_pct),
                        ('S5', route_sdi_pct)]:
            r = fn(df, pct)
            rows.append({**evaluate(df, r), 'strategy': tag,
                         'pct_l1_to_l2': pct, 'pct_l2_to_l3': 0.0})

    # 2-D sweep for S6 (two-stage)
    pct_l1_to_l2_grid = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    pct_l2_to_l3_grid = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]
    for p1 in pct_l1_to_l2_grid:
        for p2 in pct_l2_to_l3_grid:
            r = route_two_stage(df, p1, p2, llm_suffix=llm_suffix)
            rows.append({**evaluate(df, r), 'strategy': 'S6',
                         'pct_l1_to_l2': p1, 'pct_l2_to_l3': p2})

    return pd.DataFrame(rows)


def pareto_envelope(points: pd.DataFrame, x_col='cost_per_1k', y_col='f1_macro') -> pd.DataFrame:
    """Return the upper-left Pareto envelope (low x, high y)."""
    pts = points[[x_col, y_col, 'name']].sort_values(x_col).reset_index(drop=True)
    envelope = []
    best_y = -np.inf
    for _, p in pts.iterrows():
        if p[y_col] > best_y:
            envelope.append(p)
            best_y = p[y_col]
    return pd.DataFrame(envelope)


def fig_main_pareto(sweep: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Single-point baselines
    s_baseline = sweep[sweep['strategy'].isin(['S0', 'S1', 'S2'])]
    for _, p in s_baseline.iterrows():
        marker, color, label = {
            'S0': ('s', COLORS['vader'],   'Always-L1 (VADER)'),
            'S1': ('s', COLORS['finbert'], 'Always-L2 (FinBERT)'),
            'S2': ('s', COLORS['llm'],     'Always-L3 (LLM)'),
        }[p['strategy']]
        ax.scatter(p['cost_per_1k'], p['f1_macro'], s=110,
                   marker=marker, color=color, edgecolor='black',
                   linewidth=1.0, zorder=5, label=label)

    # Curves: S3, S4, S5
    for tag, color, label in [
        ('S3', '#bbbbbb', 'S3 Random escalation'),
        ('S4', '#666666', 'S4 Confidence escalation'),
        ('S5', '#1f77b4', r'S5 SDI$_\mathrm{LE}$ escalation (ours)'),
    ]:
        sub = sweep[sweep['strategy'] == tag].sort_values('cost_per_1k')
        # Take the Pareto envelope of the curve for cleanliness
        env = pareto_envelope(sub)
        ax.plot(env['cost_per_1k'], env['f1_macro'],
                marker='o', markersize=4, lw=1.4, color=color, label=label, alpha=0.95)

    # S6 two-stage — highlight as the headline curve
    sub6 = sweep[sweep['strategy'] == 'S6'].copy()
    env6 = pareto_envelope(sub6)
    ax.plot(env6['cost_per_1k'], env6['f1_macro'],
            marker='D', markersize=6, lw=2.4, color='#d62728',
            label='S6 SDI two-stage (ours, full)', zorder=4)

    ax.set_xscale('log')
    ax.set_xlabel('Inference cost (USD per 1000 sentences, log scale)')
    ax.set_ylabel('F1-Macro')
    ax.set_title('Token-Economic Pareto Frontier (FPB)')
    ax.set_ylim(0.45, 0.95)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


def operating_points_table(sweep: pd.DataFrame, llm_suffix: str | None,
                           out_path: Path) -> pd.DataFrame:
    """Pick three named operating points along S6 and the Always-L3 reference."""
    s6 = sweep[sweep['strategy'] == 'S6'].copy()
    targets = {
        'Budget':   {'pct_l1_to_l2': 0.10, 'pct_l2_to_l3': 0.01},
        'Balanced': {'pct_l1_to_l2': 0.30, 'pct_l2_to_l3': 0.05},
        'Premium':  {'pct_l1_to_l2': 0.70, 'pct_l2_to_l3': 0.25},
    }
    rows = []
    for name, cfg in targets.items():
        match = s6[(s6['pct_l1_to_l2'] == cfg['pct_l1_to_l2'])
                   & (s6['pct_l2_to_l3'] == cfg['pct_l2_to_l3'])]
        if len(match) == 1:
            r = match.iloc[0]
            rows.append({'point': name, **r[['cost_per_1k', 'f1_macro',
                                              'mean_latency_ms', 'l1_pct',
                                              'l2_pct', 'l3_pct']].to_dict()})
    # Reference Always-L3
    ref = sweep[sweep['strategy'] == 'S2'].iloc[0]
    rows.append({'point': 'Always-L3 (ref)', **ref[['cost_per_1k', 'f1_macro',
                                                     'mean_latency_ms', 'l1_pct',
                                                     'l2_pct', 'l3_pct']].to_dict()})
    out = pd.DataFrame(rows)
    print("\nOperating points (S6 with " +
          (f"Qwen-{llm_suffix or '7b'}" + " as L3):"))
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # LaTeX export
    with open(out_path, 'w') as f:
        f.write("% Auto-generated by experiments/L3_routing_pareto.py\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Point & \\$/1k & F1 & Lat (ms) & \\%L1 & \\%L2 & \\%L3 \\\\\n\\midrule\n")
        for _, r in out.iterrows():
            f.write(
                f"{r['point']} & {r['cost_per_1k']:.4f} & {r['f1_macro']:.3f} & "
                f"{r['mean_latency_ms']:.1f} & "
                f"{r['l1_pct']*100:.1f} & {r['l2_pct']*100:.1f} & {r['l3_pct']*100:.1f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  ✓ saved {out_path}")
    return out


def main(args):
    apply_style()

    src_csv = RESULTS_DATA_DIR / 'sdi_data.csv'
    if not src_csv.exists():
        raise FileNotFoundError(f"Run L2 first to produce {src_csv}")
    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} rows from {src_csv}")

    suffix = SIZE_TO_SUFFIX.get(args.llm_size or '7B')

    # If using a non-7B size, we need its cost / latency columns to exist.
    if suffix is not None:
        for c in [f'llm_{suffix}_cost_usd', f'llm_{suffix}_latency_ms', f'llm_{suffix}_label']:
            if c not in df.columns:
                raise KeyError(f"Column {c!r} missing — run L1.5 with this size first.")

    sweep = sweep_pareto(df, llm_suffix=suffix)
    suffix_tag = (args.llm_size or '7B').lower().replace('.', 'p')
    out_csv = RESULTS_DATA_DIR / f"pareto_points_{suffix_tag}.csv"
    sweep.to_csv(out_csv, index=False)
    print(f"\n✓ saved {out_csv}  ({len(sweep)} points)")

    fig_main_pareto(sweep, FIGURES_DIR / f"fig_main_pareto_{suffix_tag}.png")
    operating_points_table(sweep, suffix, TABLES_DIR / f"operating_points_{suffix_tag}.tex")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm-size', choices=list(SIZE_TO_SUFFIX.keys()), default='7B')
    args = parser.parse_args()
    main(args)
