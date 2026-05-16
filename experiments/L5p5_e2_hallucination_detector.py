"""L5.5 Experiment E2: SDI_ER as hallucination detector.

When the LLM is wrong (hallucinates a label), does FinBERT systematically
disagree more than when the LLM is right? If yes, SDI_ER serves as a
post-hoc hallucination signal — useful for production trust scoring.

Outputs:
    results/data/e2_hallucination.csv     — per-stratum SDI stats
    results/figures/fig_e2_halluc_roc.png — ROC of SDI_ER as hallucination flag
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR
from src.viz.style import apply_style, COLORS


def main():
    apply_style()
    sdi_path = RESULTS_DATA_DIR / 'sdi_data.csv'
    if not sdi_path.exists():
        # Fallback to the FPB backup if current sdi_data is the TFNS variant
        backup = RESULTS_DATA_DIR / 'sdi_data_fpb_backup2.csv'
        if backup.exists():
            sdi_path = backup
    sdi = pd.read_csv(sdi_path)
    print(f"Loaded {len(sdi)} sentences from {sdi_path}")

    # Stratify by whether the LLM is correct vs wrong
    sdi['llm_correct'] = (sdi['llm_label'] == sdi['label_text']).astype(int)
    n_right = int(sdi['llm_correct'].sum())
    n_wrong = int((1 - sdi['llm_correct']).sum())
    print(f"  LLM right on {n_right}/{len(sdi)} ({100*n_right/len(sdi):.1f}%)")
    print(f"  LLM wrong on {n_wrong}/{len(sdi)} ({100*n_wrong/len(sdi):.1f}%)")

    # Per-stratum SDI_ER stats
    rows = []
    for sdi_col in ['sdi_le', 'sdi_lr', 'sdi_er', 'sdi_max']:
        right_vals = sdi.loc[sdi['llm_correct'] == 1, sdi_col]
        wrong_vals = sdi.loc[sdi['llm_correct'] == 0, sdi_col]
        from scipy import stats
        t, p = stats.ttest_ind(wrong_vals, right_vals, equal_var=False)
        # AUC of "SDI > τ" as a flag for "LLM is wrong" (target = 1 - llm_correct)
        try:
            auc = roc_auc_score(1 - sdi['llm_correct'].to_numpy(), sdi[sdi_col].to_numpy())
        except Exception:
            auc = float('nan')
        rows.append({
            'sdi_col':       sdi_col,
            'mean_right':    float(right_vals.mean()),
            'mean_wrong':    float(wrong_vals.mean()),
            'mean_diff':     float(wrong_vals.mean() - right_vals.mean()),
            'welch_t':       float(t),
            'p_value':       float(p),
            'auc_as_halluc_flag': auc,
        })
    summary = pd.DataFrame(rows)
    print("\n=== SDI vs LLM correctness ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(RESULTS_DATA_DIR / 'e2_hallucination.csv', index=False)

    # ROC for the strongest signal (SDI_ER)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for sdi_col, color in [('sdi_le', COLORS['sdi_le']),
                            ('sdi_lr', COLORS['sdi_lr']),
                            ('sdi_er', COLORS['sdi_er']),
                            ('sdi_max', '#9467bd')]:
        try:
            fpr, tpr, _ = roc_curve(1 - sdi['llm_correct'].to_numpy(),
                                    sdi[sdi_col].to_numpy())
            auc = roc_auc_score(1 - sdi['llm_correct'].to_numpy(),
                                sdi[sdi_col].to_numpy())
            ax.plot(fpr, tpr, lw=1.6, color=color,
                    label=f"{sdi_col}  (AUC={auc:.3f})")
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], '--', color='#888', lw=0.8)
    ax.set_xlabel('False positive rate (LLM is right but flagged)')
    ax.set_ylabel('True positive rate (LLM is wrong, caught)')
    ax.set_title('SDI as a post-hoc LLM-hallucination flag')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    out = FIGURES_DIR / 'fig_e2_halluc_roc.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ saved {out}")


if __name__ == '__main__':
    main()
