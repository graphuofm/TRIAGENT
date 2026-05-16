"""Calibration + bootstrap CIs for paper rigor.

Two cheap-but-impactful additions to §5 (Experiments) that signal academic
rigor without adding new GPU work:

1. Reliability diagrams (calibration plot) per single-agent + per protocol
   showing how well each agent's reported confidence matches its actual
   accuracy. ECE (expected calibration error) per agent reported in a
   table.

2. Bootstrap 95% CIs on F1-Macro for every (single-agent, protocol)
   reported in §5. 1000 resamples, takes <30 s on CPU.

Outputs:
    results/figures/fig_calibration.png
    results/data/calibration_ece.csv
    results/data/f1_macro_bootstrap_ci.csv
    results/tables/f1_macro_with_ci.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR, TABLES_DIR
from src.viz.style import apply_style, COLORS


def expected_calibration_error(probs: np.ndarray, y_true_correct: np.ndarray,
                                n_bins: int = 10) -> tuple[float, list]:
    """Compute ECE and per-bin (mean_conf, mean_acc, count)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            bin_data.append((float((lo + hi) / 2), 0.0, 0))
            continue
        m_conf = float(probs[mask].mean())
        m_acc  = float(y_true_correct[mask].mean())
        ece += (mask.sum() / n) * abs(m_conf - m_acc)
        bin_data.append((m_conf, m_acc, int(mask.sum())))
    return float(ece), bin_data


def bootstrap_f1_macro(y_true: np.ndarray, y_pred: np.ndarray,
                        n_resamples: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_lo_2.5%, ci_hi_97.5%)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        f1s[i] = f1_score(y_true[idx], y_pred[idx], average='macro')
    point = f1_score(y_true, y_pred, average='macro')
    return float(point), float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def fig_calibration(per_agent: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (bin_data, color) in per_agent.items():
        confs = [b[0] for b in bin_data if b[2] > 0]
        accs  = [b[1] for b in bin_data if b[2] > 0]
        ax.plot(confs, accs, marker='o', lw=1.6, label=name, color=color, alpha=0.85)
    ax.plot([0, 1], [0, 1], '--', color='#888', lw=0.8, label='perfect calibration')
    ax.set_xlabel('Reported confidence')
    ax.set_ylabel('Empirical accuracy')
    ax.set_title('Calibration (reliability diagram)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


def main():
    apply_style()
    sdi = pd.read_csv(RESULTS_DATA_DIR / 'sdi_data.csv')
    print(f"Loaded {len(sdi)} sentences")

    gold = sdi['label_text'].to_numpy()

    # === Calibration: each agent's confidence vs actual accuracy ===
    print("\n=== Calibration ===")
    per_agent_bins = {}
    ece_rows = []
    for name, label_col, conf_col, color in [
        ('VADER',     'vader_label',   'vader_confidence',   COLORS['vader']),
        ('FinBERT',   'finbert_label', 'finbert_confidence', COLORS['finbert']),
        ('Qwen-7B',   'llm_label',     'llm_confidence',     COLORS['llm']),
    ]:
        if conf_col not in sdi.columns:
            continue
        correct = (sdi[label_col] == gold).astype(int).to_numpy()
        confs = sdi[conf_col].to_numpy()
        ece, bin_data = expected_calibration_error(confs, correct)
        per_agent_bins[name] = (bin_data, color)
        ece_rows.append({'agent': name, 'ECE': ece, 'mean_acc': float(correct.mean()),
                         'mean_conf': float(confs.mean())})
        print(f"  {name:<10} ECE={ece:.3f}  acc={correct.mean():.3f}  conf={confs.mean():.3f}")
    pd.DataFrame(ece_rows).to_csv(RESULTS_DATA_DIR / 'calibration_ece.csv', index=False)
    fig_calibration(per_agent_bins, FIGURES_DIR / 'fig_calibration.png')

    # === Bootstrap F1 CIs across single-agent + interaction strategies ===
    print("\n=== Bootstrap F1-Macro 95% CIs ===")
    rows = []

    # Single agents
    for name, col in [('VADER', 'vader_label'),
                       ('FinBERT', 'finbert_label'),
                       ('Qwen-0.5B', 'llm_0p5b_label'),
                       ('Qwen-1.5B', 'llm_1p5b_label'),
                       ('Qwen-3B',   'llm_3b_label'),
                       ('Qwen-7B',   'llm_label'),
                       ('Qwen-14B-4bit', 'llm_14b_label')]:
        if col not in sdi.columns:
            continue
        y_pred = sdi[col].to_numpy()
        f1, lo, hi = bootstrap_f1_macro(gold, y_pred)
        rows.append({'strategy': name, 'f1_macro': f1,
                     'ci_lo': lo, 'ci_hi': hi, 'ci_width': hi - lo})

    # Interaction protocols (read per-protocol CSVs)
    for path in sorted(RESULTS_DATA_DIR.glob('interaction_results_*.csv')):
        stem = path.stem.replace('interaction_results_', '')
        df = pd.read_csv(path)
        # Match on sentence_id
        gold_aligned = sdi.set_index('sentence_id').loc[df['sentence_id'], 'label_text'].to_numpy() \
            if 'sentence_id' in df.columns else df['label_text'].to_numpy()
        for col_name in ['vote_label', 'critic_label', 'debate_label']:
            if col_name not in df.columns:
                continue
            tag = col_name.replace('_label', '')
            f1, lo, hi = bootstrap_f1_macro(gold_aligned, df[col_name].to_numpy())
            rows.append({'strategy': f'{tag}@{stem}', 'f1_macro': f1,
                         'ci_lo': lo, 'ci_hi': hi, 'ci_width': hi - lo})

    bs = pd.DataFrame(rows)
    print(bs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    bs.to_csv(RESULTS_DATA_DIR / 'f1_macro_bootstrap_ci.csv', index=False)

    # LaTeX table
    with open(TABLES_DIR / 'f1_macro_with_ci.tex', 'w') as f:
        f.write("% Auto-generated by L5p5_calibration_bootstrap.py\n")
        f.write("\\begin{tabular}{lr}\n\\toprule\n")
        f.write("Strategy & F1-Macro [95\\% CI] \\\\\n\\midrule\n")
        for _, r in bs.iterrows():
            strat = r['strategy'].replace('_', r'\_')
            f.write(f"{strat} & {r['f1_macro']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  ✓ saved {TABLES_DIR / 'f1_macro_with_ci.tex'}")


if __name__ == '__main__':
    main()
