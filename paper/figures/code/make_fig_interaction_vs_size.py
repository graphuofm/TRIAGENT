"""Interaction-vs-size figure (the paper's central figure for §5.4).

Shows three curves vs LLM parameter count on FPB:
  * Single-agent LLM   (Qwen-N alone)
  * Critic protocol    (LLM critiques V+F outputs) — the FLAT one
  * Debate protocol    (LLM round-2 reconciliation) — the RAMP

Plus the FinBERT-alone reference horizontal line as an upper bound.
A single Mistral-7B point on each curve as the cross-family check.

Story in the figure:
  Critic plateaus at ≈0.87 across 1.5B → 7B → "interaction substitutes
  for parameters". Debate ramps from 0.69 to 0.87 → debate needs the
  LLM to be capable. The plateau is family-specific: Mistral critic
  lands lower (0.79) but Mistral debate is comparable to its critic
  (0.82).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS

# ----------------------------------------------------------------------------
# Locate the project's results/data directory regardless of cwd
# ----------------------------------------------------------------------------
ROOT = HERE.parent.parent.parent           # paper/figures/code → triagent/
RESULTS = ROOT / 'results' / 'data'


def _f1_for(suffix: str | None) -> float | None:
    """Single-agent F1 from the canonical sdi_data.csv."""
    df = pd.read_csv(ROOT / 'results' / 'data' / 'sdi_data_fpb_backup2.csv')
    col = 'llm_label' if suffix is None else f'llm_{suffix}_label'
    if col not in df.columns:
        return None
    from sklearn.metrics import f1_score
    return float(f1_score(df['label_text'], df[col], average='macro'))


def _protocol_f1(suffix: str, protocol: str) -> float | None:
    """Look up F1 across both new (proto_size) and legacy (size) naming."""
    candidates = [
        RESULTS / f'interaction_summary_{protocol}_{suffix}.csv',
        RESULTS / f'interaction_summary_{suffix}.csv',                    # legacy
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row = df[df['protocol'] == protocol]
        if len(row):
            return float(row['f1_macro'].iloc[0])
    return None


# Fallback F1 numbers extracted from /tmp/critic_7b.log (logged 2026-05-01)
# for the cases where the per-protocol summary CSV was overwritten by a
# subsequent run of a different protocol on the same size.
_KNOWN_F1_FALLBACK = {
    ('critic', '7b'):  0.8602,
    ('debate', '7b'):  0.8722,
    ('critic', '1p5b'): 0.8707,
    ('critic', '3b'):  0.8702,
}


def _resolve_f1(suffix: str, protocol: str) -> float | None:
    val = _protocol_f1(suffix, protocol)
    if val is None:
        val = _KNOWN_F1_FALLBACK.get((protocol, suffix))
    return val


def main():
    apply_paper_style()

    # ---- Single-agent F1 from the multi-size sweep ----
    qwen_sizes = [1.5, 3.0, 7.0]
    qwen_suffix = ['1p5b', '3b', None]   # None = canonical Qwen-7B columns
    single_f1 = [_f1_for(s) for s in qwen_suffix]

    # ---- Critic + debate F1 ----
    critic_suffix = ['1p5b', '3b', '7b']
    critic_f1 = [_resolve_f1(s, 'critic') for s in critic_suffix]
    debate_f1 = [_resolve_f1(s, 'debate') for s in critic_suffix]

    # ---- Mistral-7B cross-family points ----
    mistral_single = _f1_for('mistral7b')
    mistral_critic = _resolve_f1('mistral-7b', 'critic')
    mistral_debate = _resolve_f1('mistral-7b', 'debate')

    # ---- Phi-3.5-mini (3.8B, cross-family) ----
    phi_critic = _resolve_f1('phi-3p5-mini', 'critic')  # 0.857

    # ---- FinBERT reference ----
    df = pd.read_csv(ROOT / 'results' / 'data' / 'sdi_data_fpb_backup2.csv')
    from sklearn.metrics import f1_score
    finbert_f1 = float(f1_score(df['label_text'], df['finbert_label'], average='macro'))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    # Single-agent — grey, dashed (baseline)
    ax.plot(qwen_sizes, single_f1, color=PAPER_COLORS['vader'], lw=2.4,
            marker='o', markersize=11, ls='--',
            label='Single-agent LLM')

    # Critic — sky blue, FLAT line
    ax.plot(qwen_sizes, critic_f1, color=PAPER_COLORS['critic'], lw=3.0,
            marker='s', markersize=12,
            label='Committee + critic protocol  (flat: interaction $\\approx$ parameters)')

    # Debate — orange, RAMP
    ax.plot(qwen_sizes, debate_f1, color=PAPER_COLORS['debate'], lw=3.0,
            marker='D', markersize=12,
            label='Committee + debate protocol  (ramp: needs capable LLM)')

    # Mistral cross-family — outlined markers, no connecting line
    if mistral_single is not None:
        ax.plot([7.0], [mistral_single], color=PAPER_COLORS['vader'],
                marker='o', markersize=14, mfc='white', mew=2.5, ls='',
                label='Mistral-7B cross-family')
    if mistral_critic is not None:
        ax.plot([7.0], [mistral_critic], color=PAPER_COLORS['critic'],
                marker='s', markersize=15, mfc='white', mew=2.5, ls='')
    if mistral_debate is not None:
        ax.plot([7.0], [mistral_debate], color=PAPER_COLORS['debate'],
                marker='D', markersize=15, mfc='white', mew=2.5, ls='')

    # Phi-3.5-mini critic point at 3.8B (between 3B and 7B on x-axis)
    if phi_critic is not None:
        ax.plot([3.8], [phi_critic], color=PAPER_COLORS['critic'],
                marker='P', markersize=16, mfc='white', mew=2.5, ls='',
                label=f'Phi-3.5-mini critic  (cross-family, F1$=${phi_critic:.2f})')

    # FinBERT reference horizontal
    ax.axhline(finbert_f1, color=PAPER_COLORS['finbert'], lw=2.4, ls=':',
               label=f'FinBERT alone  (specialist ceiling, F1$=${finbert_f1:.2f})')

    # Annotate the plateau
    plateau_y = float(np.mean([f for f in critic_f1 if f is not None]))
    ax.annotate(f'critic plateau\n(F1 $\\approx$ {plateau_y:.2f})',
                xy=(3.0, plateau_y), xytext=(3.2, plateau_y - 0.07),
                color=PAPER_COLORS['critic'], fontsize=14, fontweight='bold',
                ha='left',
                arrowprops=dict(arrowstyle='->', color=PAPER_COLORS['critic'],
                                lw=1.6, connectionstyle='arc3,rad=0.18'))

    ax.set_xscale('log')
    ax.set_xticks(qwen_sizes)
    ax.set_xticklabels(['1.5', '3', '7'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('LLM parameter count  (B, log scale)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_ylim(0.55, 0.94)
    ax.set_xlim(1.2, 9.0)
    ax.set_title('Interaction substitutes for parameters within a model family')
    ax.legend(loc='lower right', fontsize=12, ncol=1, handletextpad=0.6)

    save_paper_figure(fig, 'fig_interaction_vs_size')


if __name__ == '__main__':
    main()
