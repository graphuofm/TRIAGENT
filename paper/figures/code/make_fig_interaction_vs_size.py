"""§5.2 Interaction vs size — the paper's central figure (v3).

Story: across 1.5B → 7B Qwen, the critic protocol plateaus at
F1 ≈ 0.87 while the single-agent LLM stays well below and debate
only catches up at 7B. Cross-family (hollow markers): Mistral-7B
critic 0.79 (plateau is family-specific); Phi-3.5-mini critic 0.86
(close to plateau).

Times serif, locked Wong palette, big fonts.
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


def _f1_for(suffix):
    df = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')
    col = 'llm_label' if suffix is None else f'llm_{suffix}_label'
    if col not in df.columns:
        return None
    from sklearn.metrics import f1_score
    return float(f1_score(df['label_text'], df[col], average='macro'))


def _protocol_f1(suffix, protocol):
    candidates = [
        RESULTS / f'interaction_summary_{protocol}_{suffix}.csv',
        RESULTS / f'interaction_summary_{suffix}.csv',
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row = df[df['protocol'] == protocol]
        if len(row):
            return float(row['f1_macro'].iloc[0])
    return None


_KNOWN_F1_FALLBACK = {
    ('critic', '7b'):   0.8602,
    ('debate', '7b'):   0.8722,
    ('critic', '1p5b'): 0.8707,
    ('critic', '3b'):   0.8702,
}


def _resolve(suffix, protocol):
    v = _protocol_f1(suffix, protocol)
    return v if v is not None else _KNOWN_F1_FALLBACK.get((protocol, suffix))


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
        'legend.fontsize':     14,
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

    qwen_sizes      = [1.5, 3.0, 7.0]
    qwen_suffix     = ['1p5b', '3b', None]
    single_f1       = [_f1_for(s) for s in qwen_suffix]

    critic_f1       = [_resolve(s, 'critic') for s in ['1p5b', '3b', '7b']]
    debate_f1       = [_resolve(s, 'debate') for s in ['1p5b', '3b', '7b']]

    mistral_single  = _f1_for('mistral7b')
    mistral_critic  = _resolve('mistral-7b', 'critic')
    mistral_debate  = _resolve('mistral-7b', 'debate')
    phi_critic      = _resolve('phi-3p5-mini', 'critic')

    # FinBERT reference
    df = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')
    from sklearn.metrics import f1_score
    finbert_f1 = float(f1_score(df['label_text'], df['finbert_label'],
                                 average='macro'))

    fig, ax = plt.subplots(figsize=(10.4, 6.4))

    # FinBERT specialist ceiling (dashed blue)
    ax.axhline(finbert_f1, color=WONG['blue'], lw=2.4, ls='--', alpha=0.85,
               label=f'FinBERT  (specialist ceiling, F1$=${finbert_f1:.2f})',
               zorder=2)

    # Single-agent LLM (grey dashed) — the baseline that's bad
    ax.plot(qwen_sizes, single_f1, color=WONG['grey'], lw=2.0,
            ls=':', marker='o', markersize=10,
            label='Qwen2.5  (single agent)', zorder=3)

    # Critic protocol — sky_blue, FLAT plateau
    ax.plot(qwen_sizes, critic_f1, color=WONG['sky_blue'], lw=3.6,
            marker='s', markersize=14,
            label=r'$+$ critic protocol  (plateau)', zorder=4)

    # Debate protocol — orange, RAMP
    ax.plot(qwen_sizes, debate_f1, color=WONG['orange'], lw=3.0,
            marker='D', markersize=12,
            label=r'$+$ debate protocol  (ramp)', zorder=4)

    # Cross-family (hollow markers, in-figure labels, no line) ──────
    # Convention: single-point reference from a different model family.
    # Don't connect with a line (no "trend"), put the name in-figure.
    if mistral_critic is not None:
        ax.plot([7.0], [mistral_critic], color=WONG['sky_blue'],
                marker='s', markersize=16, mfc='white', mew=2.4, ls='',
                zorder=5)
        ax.annotate(f'Mistral-7B critic\n(F1$=${mistral_critic:.2f})',
                    xy=(7.0, mistral_critic),
                    xytext=(7.4, mistral_critic - 0.045),
                    fontsize=13, color='black', fontweight='bold',
                    ha='left', va='center', zorder=6)
    if phi_critic is not None:
        ax.plot([3.8], [phi_critic], color=WONG['sky_blue'],
                marker='P', markersize=16, mfc='white', mew=2.4, ls='',
                zorder=5)
        ax.annotate(f'Phi-3.5-mini critic\n(F1$=${phi_critic:.2f})',
                    xy=(3.8, phi_critic),
                    xytext=(4.0, phi_critic + 0.025),
                    fontsize=13, color='black', fontweight='bold',
                    ha='left', va='bottom', zorder=6)

    # Annotate the plateau
    plateau_y = float(np.mean([f for f in critic_f1 if f is not None]))
    ax.annotate(
        f'critic plateau\nF1 $\\approx$ {plateau_y:.2f}',
        xy=(3.0, plateau_y), xytext=(2.0, 0.73),
        fontsize=15, fontweight='bold', color='black',
        ha='center', va='center',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                        connectionstyle='arc3,rad=0.20',
                        shrinkA=2, shrinkB=6),
        zorder=6,
    )

    ax.set_xscale('log')
    ax.set_xticks(qwen_sizes); ax.set_xticklabels(['1.5B', '3B', '7B'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('LLM parameter count  (log scale)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_ylim(0.55, 0.94)
    ax.set_xlim(1.2, 9.0)
    ax.set_title('Interaction substitutes for parameters')

    leg = ax.legend(loc='lower right', frameon=True, framealpha=0.94,
                    edgecolor='0.55', handletextpad=0.6, borderpad=0.55,
                    fontsize=13)
    leg.get_frame().set_linewidth(0.6)

    save_paper_figure(fig, 'fig_interaction_vs_size')


if __name__ == '__main__':
    main()
