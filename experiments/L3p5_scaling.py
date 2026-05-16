"""L3.5: Scaling-inflection + interaction-vs-size figures.

Builds three figures from the multi-size committee_data.csv (after L1.5)
and any available interaction summaries (after L2.5 critic/debate runs):

    fig_scaling_inflection.png — F1 vs LLM parameter count
                                 (with single-agent and committee curves)
    fig_pareto_multi_size.png  — cost-vs-F1 Pareto with one curve per size
    fig_interaction_vs_size.png — F1 of {single-agent, vote, critic, debate}
                                  as functions of LLM size, plus the
                                  "minimum-viable agentic" headline annotation

This script is run after L1.5 finishes and (ideally) after L2.5 has been
run for at least 2-3 sizes. Missing sizes are silently skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR
from src.viz.style import apply_style, COLORS


# size_key → (params_in_billions, suffix in committee_data columns, display_label)
SIZES = [
    ('0.5B', 0.5,  '0p5b', 'Qwen-0.5B'),
    ('1.5B', 1.5,  '1p5b', 'Qwen-1.5B'),
    ('3B',   3.0,  '3b',   'Qwen-3B'),
    ('7B',   7.0,  None,   'Qwen-7B'),  # canonical 'llm_*' columns
    ('14B',  14.0, '14b',  'Qwen-14B (4bit)'),
]


def _llm_label_col(suffix: str | None) -> str:
    return 'llm_label' if suffix is None else f'llm_{suffix}_label'


def _llm_cost_col(suffix: str | None) -> str:
    return 'llm_cost_usd' if suffix is None else f'llm_{suffix}_cost_usd'


def collect_per_size_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, params, suffix, label in SIZES:
        col = _llm_label_col(suffix)
        if col not in df.columns:
            continue
        f1 = f1_score(df['label_text'], df[col], average='macro')
        acc = accuracy_score(df['label_text'], df[col])
        cost_col = _llm_cost_col(suffix)
        cost_per_1k = (df[cost_col].sum() / len(df) * 1000) if cost_col in df.columns else float('nan')
        rows.append({'size_key': key, 'params_b': params, 'suffix': suffix or '7b',
                     'label': label, 'acc': acc, 'f1_macro': f1,
                     'cost_per_1k': cost_per_1k})
    return pd.DataFrame(rows)


def collect_interaction_summaries() -> pd.DataFrame:
    """Read every interaction_summary_*.csv in RESULTS_DATA_DIR.

    Filenames may be either the legacy `interaction_summary_<suffix>.csv`
    or the per-protocol `interaction_summary_<protocol>_<suffix>.csv`.
    Rows are de-duplicated on (size_key, protocol), preferring the most
    recently modified file when conflicts arise.
    """
    rows = []
    paths = sorted(RESULTS_DATA_DIR.glob('interaction_summary_*.csv'),
                   key=lambda p: p.stat().st_mtime)
    for path in paths:
        # Recover suffix from filename: strip prefix + ".csv", split off
        # an optional protocol prefix (vote/critic/debate).
        stem = path.stem.replace('interaction_summary_', '')
        parts = stem.split('_')
        # Match suffix to a registered size suffix (case insensitive).
        suffix_match = None
        size_match = None
        for key, params, suffix, label in SIZES:
            sfx = (suffix or '7b')
            if parts[-1].lower() == sfx.lower():
                suffix_match = sfx
                size_match = (key, params, label)
                break
        if not size_match:
            continue
        s = pd.read_csv(path)
        for _, r in s.iterrows():
            rows.append({'size_key': size_match[0], 'params_b': size_match[1],
                         'label': size_match[2],
                         'protocol': r['protocol'], 'f1_macro': r['f1_macro'],
                         'extra_cost_usd': r['extra_cost_usd']})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Keep the LAST occurrence per (size_key, protocol) — file mtime sort above
    # ensures more-recent files override older ones.
    df = df.drop_duplicates(subset=['size_key', 'protocol'], keep='last').reset_index(drop=True)
    return df


def fig_scaling_inflection(per_size: pd.DataFrame,
                           finbert_f1: float, vader_f1: float,
                           out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(per_size['params_b'], per_size['f1_macro'],
            marker='o', lw=2.0, color=COLORS['llm'], label='Qwen-N (single-agent)')
    ax.axhline(finbert_f1, ls='--', color=COLORS['finbert'],
               label=f'FinBERT (F1={finbert_f1:.3f})')
    ax.axhline(vader_f1, ls='--', color=COLORS['vader'],
               label=f'VADER (F1={vader_f1:.3f})')
    ax.set_xscale('log')
    ax.set_xlabel('LLM parameter count (B, log scale)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_title('Single-agent F1 vs LLM size')
    ax.set_ylim(0.40, 0.95)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


def fig_interaction_vs_size(per_size: pd.DataFrame,
                            interaction: pd.DataFrame,
                            finbert_f1: float,
                            out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))

    # Single-agent baseline
    ax.plot(per_size['params_b'], per_size['f1_macro'],
            marker='o', lw=2.0, color='#888888', label='Single-agent LLM')

    # Per-protocol curves (vote / critic / debate)
    if not interaction.empty:
        for proto, color in [('vote',   '#2ca02c'),
                             ('critic', '#1f77b4'),
                             ('debate', '#d62728')]:
            sub = interaction[interaction['protocol'] == proto]
            if sub.empty:
                continue
            sub = sub.sort_values('params_b')
            ax.plot(sub['params_b'], sub['f1_macro'],
                    marker='D', lw=2.0, color=color, label=f'Committee + {proto}')

    ax.axhline(finbert_f1, ls='--', color=COLORS['finbert'],
               label=f'FinBERT alone (F1={finbert_f1:.3f})')

    ax.set_xscale('log')
    ax.set_xlabel('LLM parameter count (B, log scale)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_title('Interaction vs raw scale: where does interaction substitute for parameters?')
    ax.set_ylim(0.55, 0.95)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


def fig_pareto_multi_size(per_size: pd.DataFrame,
                          finbert_f1: float, finbert_cost_per_1k: float,
                          out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(per_size['cost_per_1k'], per_size['f1_macro'],
            marker='o', lw=2.0, color=COLORS['llm'], label='Always-L3 across sizes')
    ax.scatter([finbert_cost_per_1k], [finbert_f1],
               s=120, marker='s', color=COLORS['finbert'],
               edgecolor='black', zorder=5, label='FinBERT alone')
    ax.set_xscale('log')
    ax.set_xlabel('Inference cost (USD per 1000 sentences, log scale)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_title('Multi-size cost-Pareto')
    ax.set_ylim(0.40, 0.95)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


def main(args):
    apply_style()
    # Always read from committee_data.csv for scaling figures — it always has the
    # latest LLM size columns (sdi_data.csv may be stale after L1.5 adds sizes).
    src_csv = RESULTS_DATA_DIR / 'committee_data.csv'
    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} rows from {src_csv}")

    per_size = collect_per_size_metrics(df)
    print("\nPer-size metrics:")
    print(per_size.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if per_size.empty:
        raise RuntimeError("No size columns found — run L1 / L1.5 first.")
    per_size.to_csv(RESULTS_DATA_DIR / 'scaling_per_size.csv', index=False)

    finbert_f1 = f1_score(df['label_text'], df['finbert_label'], average='macro')
    vader_f1 = f1_score(df['label_text'], df['vader_label'], average='macro')
    finbert_cost_per_1k = df['finbert_cost_usd'].sum() / len(df) * 1000

    fig_scaling_inflection(per_size, finbert_f1, vader_f1,
                           FIGURES_DIR / 'fig_scaling_inflection.png')
    fig_pareto_multi_size(per_size, finbert_f1, finbert_cost_per_1k,
                          FIGURES_DIR / 'fig_pareto_multi_size.png')

    interaction = collect_interaction_summaries()
    if not interaction.empty:
        print("\nInteraction summaries (collected from interaction_summary_*.csv):")
        print(interaction.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        fig_interaction_vs_size(per_size, interaction, finbert_f1,
                                FIGURES_DIR / 'fig_interaction_vs_size.png')
    else:
        print("\n(no interaction summaries yet — run L2.5 to populate "
              "fig_interaction_vs_size.png)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
