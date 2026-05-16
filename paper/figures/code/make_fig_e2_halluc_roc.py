"""§5.7 SDI as a post-hoc LLM-hallucination detector (v2).

ROC for "predict whether the LLM is wrong, given some SDI".
Headline: SDI_ER (Exp-Reas) reaches AUC ~ 0.90 on FPB. We render it
as the only emphasized curve; the other three SDI variants are muted
greys for context. Times serif via _style.apply_paper_style().
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    apply_paper_style()
    sdi = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')
    y = (sdi['llm_label'] != sdi['label_text']).astype(int).to_numpy()

    # Wider AND taller figure so the rotated y-axis label fits fully
    fig, ax = plt.subplots(figsize=(8.6, 7.4))

    # Context series — muted greys, thin, dashed
    context = [
        ('sdi_le',  '#B0B0B0', '--', 'SDI$_{\\mathrm{LE}}$  (Lex--Exp)'),
        ('sdi_lr',  '#8E8E8E', '-.', 'SDI$_{\\mathrm{LR}}$  (Lex--Reas)'),
        ('sdi_max', '#666666', ':',  'SDI$_{\\max}$'),
    ]
    for col, color, ls, lbl in context:
        x = sdi[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, x)
        auc = roc_auc_score(y, x)
        ax.plot(fpr, tpr, lw=1.8, color=color, ls=ls, alpha=0.85,
                label=f'{lbl}  (AUC$=${auc:.2f})', zorder=2)

    # Headline series — SDI_ER, thick vermillion (LLM/reasoner colour)
    x_er = sdi['sdi_er'].to_numpy()
    fpr_er, tpr_er, _ = roc_curve(y, x_er)
    auc_er = roc_auc_score(y, x_er)
    ax.plot(fpr_er, tpr_er, lw=4.0, color=WONG['vermillion'],
            label=f'SDI$_{{\\mathrm{{ER}}}}$  (Exp--Reas, headline)  '
                  f'(AUC$=${auc_er:.2f})',
            zorder=5)

    # Chance diagonal
    ax.plot([0, 1], [0, 1], '--', lw=1.2, color='#bbb', zorder=1)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate\n(LLM right, flagged)')
    ax.set_ylabel('True positive rate\n(LLM wrong, caught)', labelpad=8)
    ax.set_title('SDI as a post-hoc LLM hallucination detector')

    leg = ax.legend(loc='lower right', frameon=True, framealpha=0.94,
                    edgecolor='0.55', handletextpad=0.6, borderpad=0.5,
                    fontsize=13)
    leg.get_frame().set_linewidth(0.6)

    save_paper_figure(fig, 'fig_e2_halluc_roc')


if __name__ == '__main__':
    main()
