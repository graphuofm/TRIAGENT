"""§5.2 Three-way SDI decomposition (v2) — Times serif, locked Wong
palette, larger fonts, legends moved out of the way of data.

Panel 1: overlaid histograms of three pairwise SDIs.
Panel 2: scatter (SDI_LE, SDI_ER) coloured by quadrant, with dotted
         threshold lines at 0.3 and 0.7.
Panel 3: boxplot of SDI_LE by gold sentiment class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import save_paper_figure, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    mpl.rcParams.update({
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset':    'stix',
        'font.size':           15,
        'axes.titlesize':      18,
        'axes.titleweight':    'bold',
        'axes.labelsize':      17,
        'axes.labelweight':    'bold',
        'xtick.labelsize':     14,
        'ytick.labelsize':     14,
        'legend.fontsize':     13,
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

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2),
                             gridspec_kw={'wspace': 0.30})

    # ── Panel 1: three overlaid SDI histograms ──────────────────────
    ax = axes[0]
    bins = np.linspace(0, 2, 41)
    series = [
        ('sdi_le', WONG['blue'],       'SDI$_{\\mathrm{LE}}$  (Lex--Exp)'),
        ('sdi_lr', WONG['orange'],     'SDI$_{\\mathrm{LR}}$  (Lex--Reas)'),
        ('sdi_er', WONG['vermillion'], 'SDI$_{\\mathrm{ER}}$  (Exp--Reas)'),
    ]
    for col, color, lbl in series:
        # step histogram: outline only → colors don't blend in overlap zones
        ax.hist(df[col], bins=bins, label=lbl, histtype='step',
                color=color, linewidth=2.6)
    ax.set_xlabel('SDI value')
    ax.set_ylabel('Sentence count')
    ax.set_title('(a)  Three-way SDI distributions', loc='left')
    leg = ax.legend(loc='upper right', frameon=True, framealpha=0.94,
                    edgecolor='0.55', handletextpad=0.5, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)
    ax.set_xlim(0, 2)

    # ── Panel 2: 4-quadrant scatter ─────────────────────────────────
    ax = axes[1]
    quadrants = [
        ('consensus',    WONG['green']),
        ('mixed',        WONG['grey']),
        ('domain_shift', WONG['orange']),
        ('ambiguous',    WONG['purple']),
    ]
    for q, c in quadrants:
        sub = df[df['quadrant'] == q]
        ax.scatter(sub['sdi_le'], sub['sdi_er'], s=11, alpha=0.85,
                   color=c, label=f'{q.replace("_"," ")}  ($n={len(sub)}$)',
                   edgecolors='none')
    for v in (0.3,):
        ax.axvline(v, color='black', lw=1.0, ls=':')
    for v in (0.7,):
        ax.axhline(v, color='black', lw=1.0, ls=':')
    ax.text(0.31, 1.78, r'$\theta_{\mathrm{LE}}=0.3$',
            fontsize=12, color='black', va='top', ha='left')
    ax.text(1.78, 0.71, r'$\theta_{\mathrm{ER}}=0.7$',
            fontsize=12, color='black', va='bottom', ha='right')
    ax.set_xlabel('SDI$_{\\mathrm{LE}}$  (Lex--Exp)')
    ax.set_ylabel('SDI$_{\\mathrm{ER}}$  (Exp--Reas)')
    ax.set_title('(b)  Four-quadrant decomposition', loc='left')
    leg = ax.legend(loc='upper right', frameon=True, framealpha=0.94,
                    edgecolor='0.55', handletextpad=0.4, borderpad=0.45,
                    markerscale=2.2, fontsize=12)
    leg.get_frame().set_linewidth(0.6)
    ax.set_xlim(0, 1.85); ax.set_ylim(0, 1.85)

    # ── Panel 3: SDI_LE per gold class ──────────────────────────────
    ax = axes[2]
    classes = ['negative', 'neutral', 'positive']
    colors  = [WONG['vermillion'], WONG['grey'], WONG['green']]
    data = [df.loc[df['label_text'] == c, 'sdi_le'].values for c in classes]
    bp = ax.boxplot(data, tick_labels=classes, patch_artist=True,
                    widths=0.6,
                    medianprops={'color': 'black', 'linewidth': 2.0},
                    flierprops={'marker': 'o', 'markersize': 3,
                                'markerfacecolor': 'none',
                                'markeredgecolor': '0.45',
                                'alpha': 0.5})
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85)
        patch.set_edgecolor('black'); patch.set_linewidth(1.1)
    # Annotate means above each box
    for i, vals in enumerate(data):
        m = float(np.mean(vals))
        ax.text(i + 1, 1.78, f'mean$=${m:.2f}',
                fontsize=12, ha='center', va='top', color='black',
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec='0.55', lw=0.5))
    ax.set_ylabel('SDI$_{\\mathrm{LE}}$')
    ax.set_xlabel('Gold sentiment class')
    ax.set_title('(c)  SDI$_{\\mathrm{LE}}$ by gold class', loc='left')
    ax.set_ylim(0, 1.85)

    save_paper_figure(fig, 'fig_sdi_three_way')


if __name__ == '__main__':
    main()
