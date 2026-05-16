"""Per-class F1 breakdown across all strategies/sizes.

Headline F1-macro hides class-level structure: on FPB the corpus is
59% neutral / 28% positive / 13% negative, and the negative class is
where VADER famously fails. We want to know: does the committee fix
the negative class, or just the easy ones?

Reads:
    sdi_data.csv             — base committee + SDI
    interaction_results_*.csv — per-protocol final labels (any size)

Writes:
    results/data/per_class_f1.csv
    results/figures/fig_per_class_f1.png   (heatmap-ish bar chart)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR
from src.viz.style import apply_style, COLORS


CLASSES = ['negative', 'neutral', 'positive']


def per_class(df: pd.DataFrame, label_col: str) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        df['label_text'], df[label_col], labels=CLASSES, zero_division=0)
    return {
        **{f'f1_{c}':        round(float(f[i]), 4) for i, c in enumerate(CLASSES)},
        **{f'precision_{c}': round(float(p[i]), 4) for i, c in enumerate(CLASSES)},
        **{f'recall_{c}':    round(float(r[i]), 4) for i, c in enumerate(CLASSES)},
        'f1_macro': round(float(f.mean()), 4),
    }


def main():
    apply_style()
    base = pd.read_csv(RESULTS_DATA_DIR / 'sdi_data.csv')
    print(f"Loaded base sdi_data.csv: {len(base)} rows")

    rows = []

    # Single-agent baselines from base csv
    for name, col in [
        ('VADER',           'vader_label'),
        ('FinBERT',         'finbert_label'),
        ('Qwen-7B alone',   'llm_label'),
    ]:
        rows.append({'strategy': name, **per_class(base, col)})

    # Multi-size single-agent
    for size_suffix, label in [('0p5b', 'Qwen-0.5B alone'),
                                ('1p5b', 'Qwen-1.5B alone'),
                                ('3b',   'Qwen-3B alone')]:
        col = f'llm_{size_suffix}_label'
        if col in base.columns:
            rows.append({'strategy': label, **per_class(base, col)})

    # Interaction results — discover all per-protocol files
    for path in sorted(RESULTS_DATA_DIR.glob('interaction_results_*.csv')):
        stem = path.stem.replace('interaction_results_', '')
        # stem looks like 'critic_1p5b' or '1p5b' (legacy)
        if '_' in stem:
            proto, suffix = stem.rsplit('_', 1)
        else:
            proto, suffix = 'unknown', stem
        df = pd.read_csv(path)
        for col_name in ['vote_label', 'critic_label', 'debate_label']:
            if col_name not in df.columns:
                continue
            tag = col_name.replace('_label', '')
            rows.append({
                'strategy': f'{tag}@Qwen-{suffix.upper().replace("P", ".")}',
                **per_class(df, col_name),
            })

    out = pd.DataFrame(rows).drop_duplicates(subset=['strategy'])
    print("\nPer-class F1 breakdown:")
    print(out[['strategy', 'f1_negative', 'f1_neutral', 'f1_positive', 'f1_macro']]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out.to_csv(RESULTS_DATA_DIR / 'per_class_f1.csv', index=False)

    # Plot: stacked bar of per-class F1 per strategy, sorted by f1_negative
    out_sorted = out.sort_values('f1_negative')
    fig, ax = plt.subplots(figsize=(10, max(4, len(out_sorted) * 0.32)))
    y = np.arange(len(out_sorted))
    bar_h = 0.27
    ax.barh(y - bar_h, out_sorted['f1_negative'], height=bar_h,
            color='#d62728', label='F1 (negative)', alpha=0.85)
    ax.barh(y,         out_sorted['f1_neutral'],  height=bar_h,
            color='#888888', label='F1 (neutral)',  alpha=0.85)
    ax.barh(y + bar_h, out_sorted['f1_positive'], height=bar_h,
            color='#2ca02c', label='F1 (positive)', alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(out_sorted['strategy'])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('F1 per class')
    ax.set_title('Per-class F1 — sorted by F1 on the hard (negative) class')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    out_path = FIGURES_DIR / 'fig_per_class_f1.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n✓ saved {out_path}")


if __name__ == '__main__':
    main()
