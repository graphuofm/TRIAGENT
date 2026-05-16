"""TriAgent system architecture (v6) — Return-cached narrower; arrow
labels no longer overlap boxes; write-back caption sits BELOW the
Return-cached box.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import save_paper_figure, WONG


def box(ax, x, y, w, h, text, color, fontsize=17, text_color='white'):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        facecolor=color, edgecolor='black', lw=1.4, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center',
            fontsize=fontsize, color=text_color,
            weight='bold', zorder=4)


def varrow(ax, x, y_top, y_bot, lw=2.2, label=None, label_xoffset=0.55):
    """Vertical arrow with optional label placed to the right of the
    arrow (clear of the boxes above/below)."""
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=lw,
                                mutation_scale=22),
                zorder=2)
    if label:
        midy = (y_top + y_bot) / 2
        ax.text(x + label_xoffset, midy, label, fontsize=15,
                color='black', va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none'),
                zorder=4)


def main():
    mpl.rcParams.update({
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset':    'stix',
        'pdf.fonttype':        42,
        'ps.fonttype':         42,
        'figure.dpi':          120,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.04,
    })

    W, H = 14.0, 9.8
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_axis_off()

    cx_main = 4.8
    cx_side = 11.2
    bw = 6.6
    bw_side = 4.0        # narrower
    bh = 1.0
    bh_side = 1.0        # same height

    # --- Vertical positions (extra spacing → arrow labels have room) -
    y_query  = 8.55
    y_scd    = 6.95
    y_comm_b = 4.95         # committee box BOTTOM
    h_comm   = 1.45
    y_sdi    = 3.30
    y_inter  = 1.65
    y_final  = 0.10

    # ── Boxes ────────────────────────────────────────────────────────
    box(ax, cx_main - bw / 2, y_query, bw, bh,
        'Query sentence', WONG['sky_blue'], fontsize=18, text_color='black')

    box(ax, cx_main - bw / 2, y_scd, bw, bh,
        'SCD lookup  (sentence-BERT $k$-NN)',
        WONG['green'], fontsize=16)

    box(ax, cx_main - bw / 2, y_comm_b, bw, h_comm,
        'Three-tier committee\nL1 VADER  $\\to$  L2 FinBERT  $\\to$  L3 Qwen-$N$',
        WONG['blue'], fontsize=15)

    box(ax, cx_main - bw / 2, y_sdi, bw, bh,
        'SDI 3-way  +  4-quadrant routing',
        WONG['purple'], fontsize=16)

    box(ax, cx_main - bw / 2, y_inter, bw, bh,
        'Interaction:  vote  /  critic  /  debate',
        WONG['orange'], fontsize=16)

    box(ax, cx_main - bw / 2, y_final, bw, bh,
        'Final label  +  write back to SCD',
        WONG['vermillion'], fontsize=16)

    # ── Right column: Return cached label (narrower, height same) ────
    y_cached = 5.55
    box(ax, cx_side - bw_side / 2, y_cached, bw_side, bh_side,
        'Return cached label', WONG['green'], fontsize=16)

    # write back caption — directly BELOW the Return-cached box
    ax.text(cx_side, y_cached - 0.45,
            'write back (populates cache)',
            fontsize=13, color='black', ha='center', va='top',
            style='italic')

    # ── Main vertical arrows ─────────────────────────────────────────
    varrow(ax, cx_main, y_query,       y_scd + bh)                 # Query → SCD
    varrow(ax, cx_main, y_scd,         y_comm_b + h_comm,
           label='miss  ($\\sigma < \\tau$)')                       # SCD → Committee
    varrow(ax, cx_main, y_comm_b,      y_sdi + bh)                 # Committee → SDI
    varrow(ax, cx_main, y_sdi,         y_inter + bh,
           label='SDI  $> \\theta$')                                # SDI → Interaction
    varrow(ax, cx_main, y_inter,       y_final + bh)               # Interaction → Final

    # ── Side arrow: SCD → Return cached (hit) ────────────────────────
    ax.annotate('', xy=(cx_side - bw_side / 2,   y_cached + bh_side / 2),
                xytext=(cx_main + bw / 2,        y_scd + bh / 2),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=2.2,
                                mutation_scale=22), zorder=2)
    ax.text((cx_main + bw / 2 + cx_side - bw_side / 2) / 2,
            (y_scd + bh / 2 + y_cached + bh_side / 2) / 2 + 0.25,
            'hit  ($\\sigma \\geq \\tau$)',
            fontsize=15, color='black', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none'),
            zorder=4)

    # ── Side arrow: Return cached → Final label (skips committee) ────
    ax.annotate('', xy=(cx_main + bw / 2 - 0.1, y_final + bh / 2),
                xytext=(cx_side - bw_side / 2,  y_cached),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6,
                                mutation_scale=20,
                                connectionstyle='arc3,rad=-0.18'),
                zorder=1)

    # ── Write-back arrow: Final → SCD (dotted) ───────────────────────
    ax.annotate('', xy=(cx_side - bw_side / 2,  y_scd + bh / 2 + 0.15),
                xytext=(cx_main + bw / 2 - 0.3, y_final + 0.30),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.4,
                                ls=(0, (4, 3)),
                                mutation_scale=18,
                                connectionstyle='arc3,rad=0.32'),
                zorder=1)

    save_paper_figure(fig, 'fig_architecture')


if __name__ == '__main__':
    main()
