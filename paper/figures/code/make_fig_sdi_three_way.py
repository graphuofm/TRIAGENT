"""§5.2 Three-way SDI decomposition (3 panels)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    apply_paper_style(font_scale=0.92)
    df = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel 1: SDI distributions overlaid
    ax = axes[0]
    bins = np.linspace(0, 2, 41)
    series = [
        ('sdi_le', WONG['blue'],     'SDI$_{\\mathrm{LE}}$  (Lex--Exp)'),
        ('sdi_lr', WONG['purple'],   'SDI$_{\\mathrm{LR}}$  (Lex--Reas)'),
        ('sdi_er', WONG['vermillion'],'SDI$_{\\mathrm{ER}}$  (Exp--Reas)'),
    ]
    for col, color, lbl in series:
        ax.hist(df[col], bins=bins, alpha=0.55, label=lbl, color=color)
    ax.set_xlabel('SDI value')
    ax.set_ylabel('count')
    ax.set_title('Three-way SDI distributions')
    ax.legend(fontsize=11)

    # Panel 2: scatter SDI_LE vs SDI_ER, colored by quadrant
    ax = axes[1]
    quadrant_color = {
        'consensus':    WONG['green'],
        'mixed':        WONG['grey'],
        'domain_shift': WONG['orange'],
        'ambiguous':    WONG['purple'],
    }
    for q, c in quadrant_color.items():
        sub = df[df['quadrant'] == q]
        ax.scatter(sub['sdi_le'], sub['sdi_er'], s=8, alpha=0.45,
                   color=c, label=f'{q} ({len(sub)})')
    for v in (0.3, 0.7):
        ax.axvline(v, color='k', lw=0.8, ls=':')
        ax.axhline(v, color='k', lw=0.8, ls=':')
    ax.set_xlabel('SDI$_{\\mathrm{LE}}$')
    ax.set_ylabel('SDI$_{\\mathrm{ER}}$')
    ax.set_title('Four-quadrant decomposition')
    ax.legend(fontsize=10, loc='upper right', markerscale=2.0)

    # Panel 3: SDI_LE per gold class
    ax = axes[2]
    classes = ['negative', 'neutral', 'positive']
    data = [df.loc[df['label_text'] == c, 'sdi_le'].values for c in classes]
    bp = ax.boxplot(data, tick_labels=classes, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, c in zip(bp['boxes'],
                         [WONG['vermillion'], WONG['grey'], WONG['green']]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel('SDI$_{\\mathrm{LE}}$')
    ax.set_title('SDI$_{\\mathrm{LE}}$ by gold sentiment class')

    save_paper_figure(fig, 'fig_sdi_three_way')


if __name__ == '__main__':
    main()
