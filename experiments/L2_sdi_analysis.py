"""L2: Three-way SDI + Bias Diversity.

Reads `results/data/committee_data.csv` (from L1) and produces:
    - results/data/sdi_data.csv          (committee + SDI columns)
    - results/figures/fig_sdi_three_way.png
    - results/figures/fig_bias_diversity.png
    - results/tables/quadrant_summary.tex
plus printed ANOVA + pairwise t-test diagnostics.

Run:
    python experiments/L2_sdi_analysis.py
    python experiments/L2_sdi_analysis.py --llm-col llm_3b_score   # use 3B as the L3 reasoner

This experiment uses ONLY the columns produced by L1 — no GPU, no API.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR, TABLES_DIR
from src.metrics.sdi import add_sdi_columns
from src.metrics.bias_diversity import (
    pairwise_kappa,
    add_disagreement_entropy,
    error_set_overlap,
)
from src.viz.style import apply_style, COLORS, QUADRANT_ORDER


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d for two independent samples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def anova_and_pairwise(df: pd.DataFrame, sdi_col: str = 'sdi_le') -> None:
    classes = ['negative', 'neutral', 'positive']
    groups = [df.loc[df['label_text'] == c, sdi_col].values for c in classes]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"\n[{sdi_col}]  one-way ANOVA across gold classes: F={f_stat:.2f}  p={p_value:.2e}")
    for c, g in zip(classes, groups):
        print(f"  {c:>8}: n={len(g):>4}  mean={g.mean():.3f}  std={g.std():.3f}")
    print(f"\n[{sdi_col}]  pairwise Welch's t-tests + Cohen's d:")
    for i, ci in enumerate(classes):
        for cj in classes[i + 1:]:
            a = df.loc[df['label_text'] == ci, sdi_col].values
            b = df.loc[df['label_text'] == cj, sdi_col].values
            t, p = stats.ttest_ind(a, b, equal_var=False)
            d = cohens_d(a, b)
            print(f"  {ci:>8} vs {cj:<8}: t={t:+.2f}  p={p:.2e}  Cohen's d={d:+.3f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_sdi_three_way(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # Panel 1: histograms of the three SDIs overlaid
    ax = axes[0]
    bins = np.linspace(0, 2, 41)
    for col, name, color in [
        ('sdi_le', r'SDI$_\mathrm{LE}$ (Lex–Exp)',  COLORS['sdi_le']),
        ('sdi_lr', r'SDI$_\mathrm{LR}$ (Lex–Reas)', COLORS['sdi_lr']),
        ('sdi_er', r'SDI$_\mathrm{ER}$ (Exp–Reas)', COLORS['sdi_er']),
    ]:
        ax.hist(df[col], bins=bins, alpha=0.45, label=name, color=color)
    ax.set_xlabel('SDI value')
    ax.set_ylabel('count')
    ax.set_title('Three-way SDI distributions')
    ax.legend()

    # Panel 2: SDI_LE vs SDI_ER, colored by quadrant
    ax = axes[1]
    for q in QUADRANT_ORDER:
        sub = df[df['quadrant'] == q]
        ax.scatter(sub['sdi_le'], sub['sdi_er'], s=6, alpha=0.5,
                   color=COLORS[q], label=f'{q} (n={len(sub)})')
    from src.config import SDI_HIGH, SDI_LOW
    for v in (SDI_LOW, SDI_HIGH):
        ax.axvline(v, color='k', lw=0.6, ls=':')
        ax.axhline(v, color='k', lw=0.6, ls=':')
    ax.set_xlabel(r'SDI$_\mathrm{LE}$')
    ax.set_ylabel(r'SDI$_\mathrm{ER}$')
    ax.set_title('Four-quadrant decomposition')
    ax.legend(loc='upper right', fontsize=7, markerscale=2)

    # Panel 3: SDI_LE distribution per gold class (boxplot)
    ax = axes[2]
    classes = ['negative', 'neutral', 'positive']
    data = [df.loc[df['label_text'] == c, 'sdi_le'].values for c in classes]
    bp = ax.boxplot(data, tick_labels=classes, patch_artist=True,
                    medianprops={'color': 'black'})
    for patch, c in zip(bp['boxes'], ['#d62728', '#888888', '#2ca02c']):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.set_ylabel(r'SDI$_\mathrm{LE}$')
    ax.set_title('SDI$_\\mathrm{LE}$ by gold sentiment')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


def fig_bias_diversity(df: pd.DataFrame, label_cols: list[str], out_path: Path) -> None:
    K = pairwise_kappa(df, label_cols)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    ax = axes[0]
    sns.heatmap(
        K, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='YlGnBu',
        cbar_kws={'label': "Cohen's κ"}, ax=ax, square=True
    )
    short = {c: c.replace('_label', '') for c in label_cols}
    ax.set_xticklabels([short[c] for c in label_cols], rotation=0)
    ax.set_yticklabels([short[c] for c in label_cols], rotation=0)
    ax.set_title("Pairwise Cohen's κ\n(low = high diversity)")

    ax = axes[1]
    ent_bins = np.linspace(0, np.log2(3) + 1e-3, 25)
    ax.hist(df['disagreement_entropy'], bins=ent_bins, color='#1f77b4', alpha=0.75)
    ax.set_xlabel('per-sample disagreement entropy (bits)')
    ax.set_ylabel('count')
    ax.set_title('Committee disagreement entropy')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


# ---------------------------------------------------------------------------
# Quadrant summary table
# ---------------------------------------------------------------------------

def quadrant_summary_table(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    for q in QUADRANT_ORDER:
        sub = df[df['quadrant'] == q]
        if len(sub) == 0:
            continue
        gold_mix = (sub['label_text'].value_counts(normalize=True) * 100).round(1).to_dict()
        rows.append({
            'quadrant':         q,
            'n':                len(sub),
            'pct':              100 * len(sub) / n_total,
            'pct_pos':          gold_mix.get('positive', 0.0),
            'pct_neu':          gold_mix.get('neutral', 0.0),
            'pct_neg':          gold_mix.get('negative', 0.0),
            'vader_acc':        100 * accuracy_score(sub['label_text'], sub['vader_label']),
            'finbert_acc':      100 * accuracy_score(sub['label_text'], sub['finbert_label']),
            'llm_acc':          100 * accuracy_score(sub['label_text'], sub['llm_label']),
            'mean_sdi_le':      sub['sdi_le'].mean(),
            'mean_sdi_er':      sub['sdi_er'].mean(),
        })
    summary = pd.DataFrame(rows)
    print("\nQuadrant summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # LaTeX export — booktabs style, ready for the paper
    with open(out_path, 'w') as f:
        f.write("% Auto-generated by experiments/L2_sdi_analysis.py — do not edit\n")
        f.write("\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n")
        f.write("Quadrant & $n$ & \\% & \\%pos & \\%neu & \\%neg & "
                "$\\mathrm{Acc}_{\\mathrm{V}}$ & $\\mathrm{Acc}_{\\mathrm{F}}$ & "
                "$\\mathrm{Acc}_{\\mathrm{L}}$ & "
                "$\\overline{\\mathrm{SDI}_{\\mathrm{LE}}}$ & "
                "$\\overline{\\mathrm{SDI}_{\\mathrm{ER}}}$ \\\\\n\\midrule\n")
        for _, r in summary.iterrows():
            qname = r['quadrant'].replace('_', r'\_')
            f.write(
                f"{qname} & {int(r['n'])} & "
                f"{r['pct']:.1f} & {r['pct_pos']:.1f} & {r['pct_neu']:.1f} & {r['pct_neg']:.1f} & "
                f"{r['vader_acc']:.1f} & {r['finbert_acc']:.1f} & {r['llm_acc']:.1f} & "
                f"{r['mean_sdi_le']:.2f} & {r['mean_sdi_er']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  ✓ saved {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(args):
    apply_style()

    src_csv = RESULTS_DATA_DIR / "committee_data.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Run L1 first to produce {src_csv}")
    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} rows from {src_csv}")

    # Allow swapping which column plays the L3 'reasoner' role (e.g. for the
    # multi-size sweep, you can re-run L2 with --llm-col llm_3b_score etc.)
    llm_score_col = args.llm_col
    llm_label_col = llm_score_col.replace('_score', '_label')
    if llm_score_col not in df.columns:
        raise KeyError(f"{llm_score_col!r} not in committee_data.csv. "
                       f"Did you run L1 (or L1.5 for multi-size)?")
    print(f"Using L3 reasoner column: {llm_score_col} / {llm_label_col}")

    df = add_sdi_columns(df, llm_col=llm_score_col)

    # Stash a unified `llm_label` for downstream code that expects it
    if llm_label_col != 'llm_label':
        df['llm_label'] = df[llm_label_col]

    label_cols = ['vader_label', 'finbert_label', 'llm_label']
    df = add_disagreement_entropy(df, label_cols)

    # ANOVA + pairwise t-tests on SDI_LE
    anova_and_pairwise(df, 'sdi_le')
    print()
    anova_and_pairwise(df, 'sdi_er')

    # Pairwise kappa diagnostic
    K = pairwise_kappa(df, label_cols)
    print("\nPairwise Cohen's kappa:")
    print(K.round(3).to_string())

    # Error-set overlap (for §5 security / bias subsection)
    J = error_set_overlap(df, label_cols, gold_col='label_text')
    print("\nPairwise error-set Jaccard overlap (low = orthogonal failures = good for ensembling):")
    print(J.round(3).to_string())

    # Save figures
    fig_sdi_three_way(df, FIGURES_DIR / "fig_sdi_three_way.png")
    fig_bias_diversity(df, label_cols, FIGURES_DIR / "fig_bias_diversity.png")

    # Save quadrant table
    quadrant_summary_table(df, TABLES_DIR / "quadrant_summary.tex")

    # Persist augmented dataframe for L3+
    out_csv = RESULTS_DATA_DIR / "sdi_data.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ saved {out_csv}")

    # Spec checkpoints (printed for human verification)
    print("\n=== L2 checkpoint diagnostics ===")
    neg_mean = df.loc[df['label_text'] == 'negative', 'sdi_le'].mean()
    print(f"  neg-class mean SDI_LE = {neg_mean:.3f}  (spec target ≈ 0.945)")
    qcounts = df['quadrant'].value_counts(normalize=True).to_dict()
    print(f"  quadrant fractions    = "
          + ", ".join(f"{q}={qcounts.get(q, 0)*100:.1f}%" for q in QUADRANT_ORDER))
    cons = df[df['quadrant'] == 'consensus']
    dom = df[df['quadrant'] == 'domain_shift']
    if len(cons) > 0:
        print(f"  VADER acc in consensus    = {accuracy_score(cons['label_text'], cons['vader_label'])*100:.1f}% (target >90%)")
    if len(dom) > 0:
        print(f"  VADER acc in domain_shift = {accuracy_score(dom['label_text'], dom['vader_label'])*100:.1f}% (target <50%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm-col', default='llm_score',
                        help='Score column to use as the L3 reasoner '
                             '(default: llm_score = Qwen-7B). '
                             'Try llm_0p5b_score / llm_1p5b_score / llm_3b_score / llm_14b_score.')
    args = parser.parse_args()
    main(args)
