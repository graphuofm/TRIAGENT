"""Figure 1 — TriAgent system architecture (v3, no overflow).

The trick: labels short enough to FIT in the boxes at the chosen font
size. Long descriptions go in the figure caption in the .tex file
instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS, WONG


def box(ax, x, y, w, h, text, color, fontsize=14, text_color='white'):
    rect = patches.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.06,rounding_size=0.10",
                                   facecolor=color, edgecolor='black', lw=1.6,
                                   zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, weight='bold', zorder=4)


def varrow(ax, x, y_top, y_bot, color='black', lw=2.4, label=None,
           label_color=None):
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=22),
                zorder=2)
    if label:
        midy = (y_top + y_bot) / 2
        ax.text(x + 0.20, midy, label, fontsize=12,
                color=(label_color or color), va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'),
                zorder=4)


def main():
    apply_paper_style()
    W, H = 14.0, 9.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_axis_off()

    cx_main = 4.8       # main column centre
    cx_side = 11.0      # right column centre
    bw = 6.4            # main box width — wider so labels fit
    bw_side = 5.0
    bh = 0.95

    # 1) Query
    box(ax, cx_main - bw / 2, 8.5, bw, bh, 'Query sentence',
        WONG['black'], fontsize=15)

    # 2) SCD
    box(ax, cx_main - bw / 2, 7.0, bw, bh,
        'SCD lookup  (sentence-BERT $k$-NN)',
        WONG['green'], fontsize=14)
    varrow(ax, cx_main, 8.5, 7.95)

    # 3a) Three-tier committee
    box(ax, cx_main - bw / 2, 5.1, bw, 1.4,
        'Three-tier committee\n L1 VADER $\\to$ L2 FinBERT $\\to$ L3 Qwen-$N$',
        PAPER_COLORS['finbert'], fontsize=13)
    varrow(ax, cx_main, 7.0, 6.5,
           color=WONG['vermillion'], label='miss  ($<\\tau$)',
           label_color=WONG['vermillion'])

    # 3b) Cached fast-path on the right column
    box(ax, cx_side - bw_side / 2, 5.5, bw_side, bh,
        'Return cached label', WONG['green'], fontsize=14)
    # diagonal arrow from SCD (right edge) → cached fast path
    ax.annotate('', xy=(cx_side - bw_side / 2 + 0.0, 6.0),
                xytext=(cx_main + bw / 2, 7.4),
                arrowprops=dict(arrowstyle='-|>', color=WONG['green'], lw=2.4,
                                mutation_scale=22), zorder=2)
    ax.text(cx_main + bw / 2 + 1.0, 6.95, 'hit  ($\\geq\\tau$)',
            fontsize=12, color=WONG['green'], ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'), zorder=4)

    # 4) SDI module
    box(ax, cx_main - bw / 2, 3.5, bw, bh,
        'SDI 3-way  +  4-quadrant routing',
        WONG['purple'], fontsize=14)
    varrow(ax, cx_main, 5.1, 4.45)

    # 5) Interaction protocol
    box(ax, cx_main - bw / 2, 1.9, bw, bh,
        'Interaction:  vote / critic / debate',
        PAPER_COLORS['debate'], fontsize=14)
    varrow(ax, cx_main, 3.5, 2.85, label='SDI $> \\tau$',
           color=PAPER_COLORS['debate'])

    # 6) Final label
    box(ax, cx_main - bw / 2, 0.3, bw, bh,
        'Final label  +  write back to SCD',
        WONG['black'], fontsize=14)
    varrow(ax, cx_main, 1.9, 1.25)

    # 7) Fast path → final label (curved)
    ax.annotate('', xy=(cx_main + bw / 2 - 0.1, 0.95),
                xytext=(cx_side - bw_side / 2, 5.55),
                arrowprops=dict(arrowstyle='-|>', color='#888', lw=1.8,
                                mutation_scale=20,
                                connectionstyle='arc3,rad=-0.18'), zorder=1)

    # 8) Write-back from final label up to SCD (dotted green, curved)
    ax.annotate('', xy=(cx_side - bw_side / 2, 7.45),
                xytext=(cx_main + bw / 2 - 0.3, 0.5),
                arrowprops=dict(arrowstyle='-|>', color=WONG['green'], lw=1.6,
                                ls=(0, (4, 3)), mutation_scale=18,
                                connectionstyle='arc3,rad=0.32'), zorder=1)
    ax.text(cx_side + 1.4, 4.2, 'write back\n(populates cache)',
            fontsize=11, color=WONG['green'], ha='center', va='center',
            style='italic')

    ax.set_title('TriAgent  —  three-granularity committee with SDI-gated interaction and SCD cache',
                 fontsize=18, fontweight='bold', pad=12)
    save_paper_figure(fig, 'fig_architecture')


if __name__ == '__main__':
    main()
