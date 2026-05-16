"""§5.2 Single-agent F1 vs Qwen size (v3) — Times serif, locked Wong
palette, bigger fonts, the 3B dip is highlighted with an annotation.

Story: scaling is non-monotone within Qwen2.5-Instruct on FPB; the
3B dip motivates "interaction substitutes for parameters" in §5.2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import save_paper_figure, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def _f1(df, col):
    return f1_score(df['label_text'], df[col], average='macro')


def main():
    mpl.rcParams.update({
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset':    'stix',
        'font.size':           17,
        'axes.titlesize':      21,
        'axes.titleweight':    'bold',
        'axes.labelsize':      19,
        'axes.labelweight':    'bold',
        'xtick.labelsize':     16,
        'ytick.labelsize':     16,
        'legend.fontsize':     15,
        'pdf.fonttype':        42,
        'ps.fonttype':         42,
        'figure.dpi':          120,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.04,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'axes.linewidth':      1.3,
        'axes.grid':           True,
        'grid.linestyle':      '--',
        'grid.alpha':          0.25,
        'grid.linewidth':      0.6,
    })

    df = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')

    sizes_b, f1s, labels = [], [], []
    for size, suffix in [(0.5, '0p5b'), (1.5, '1p5b'),
                         (3.0, '3b'), (7.0, None)]:
        col = 'llm_label' if suffix is None else f'llm_{suffix}_label'
        if col not in df.columns:
            continue
        sizes_b.append(size); f1s.append(_f1(df, col))
        labels.append(f'{size}B')

    finbert_f1 = _f1(df, 'finbert_label')
    vader_f1   = _f1(df, 'vader_label')

    fig, ax = plt.subplots(figsize=(10.0, 6.2))

    # Shaded band: "FinBERT specialist ceiling — the gap that scaling
    # cannot close" — sits above the 7B Qwen point
    ax.axhspan(f1s[-1], finbert_f1, color=WONG['blue'], alpha=0.10,
               zorder=1)
    ax.text(0.55, (f1s[-1] + finbert_f1) / 2,
            'gap that scaling cannot close\n(7B still $-$%dpp vs. FinBERT)'
            % round((finbert_f1 - f1s[-1]) * 100),
            fontsize=14, color=WONG['blue'], fontweight='bold',
            ha='left', va='center', style='italic')

    # FinBERT reference line — dashed, blue
    ax.axhline(finbert_f1, ls='--', lw=2.4, color=WONG['blue'], alpha=0.9,
               label=f'FinBERT  (specialist, F1$=${finbert_f1:.2f})',
               zorder=2)

    # VADER reference line — dotted, grey
    ax.axhline(vader_f1, ls=':', lw=2.0, color=WONG['grey'], alpha=0.85,
               label=f'VADER  (lexicon, F1$=${vader_f1:.2f})',
               zorder=2)

    # Qwen single-agent — solid, vermillion
    ax.plot(sizes_b, f1s, marker='o', markersize=13, lw=3.2,
            color=WONG['vermillion'],
            label='Qwen2.5-Instruct  (single agent)',
            zorder=3)

    # Annotate the 3B dip — direct label above the 3B point, no arrow
    dip_x, dip_y = 3.0, f1s[2]
    ax.text(dip_x, dip_y + 0.045, '3B dip  (3B $<$ 1.5B)',
            fontsize=14, fontweight='bold', color='#CC0000',
            ha='center', va='bottom', zorder=4,
            bbox=dict(boxstyle='round,pad=0.18', fc='white',
                      ec='#CC0000', lw=0.8))

    ax.set_xscale('log')
    ax.set_xticks(sizes_b); ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('Qwen parameter count  (log scale)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_ylim(0.40, 0.95)
    ax.set_xlim(0.4, 9.5)
    ax.set_title('Scaling Qwen alone cannot beat the specialist')

    leg = ax.legend(loc='lower right', frameon=True, framealpha=0.94,
                    edgecolor='0.55', handletextpad=0.6, borderpad=0.5,
                    fontsize=14)
    leg.get_frame().set_linewidth(0.6)

    save_paper_figure(fig, 'fig_scaling_inflection')


if __name__ == '__main__':
    main()
