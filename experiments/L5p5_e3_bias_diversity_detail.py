"""L5.5 Experiment E3: detailed bias-diversity quantification.

Extends L2's pairwise-error Jaccard with:
  - Per-class confusion matrices for each agent + protocol
  - Pairwise error-set Jaccard between (V, F, L) AND between protocols
  - Bias-cancellation effect: how much does ensembling reduce minority-class
    FN rate vs the best single agent?

Outputs:
    results/data/e3_bias_overlap.csv   — pairwise Jaccard + per-class F1
    results/data/e3_minority_class_fnr.csv — minority-class recall lift
    results/figures/fig_e3_bias_overlap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR
from src.metrics.bias_diversity import error_set_overlap
from src.viz.style import apply_style


CLASSES = ['negative', 'neutral', 'positive']


def main():
    apply_style()
    sdi_path = RESULTS_DATA_DIR / 'sdi_data.csv'
    if not sdi_path.exists():
        backup = RESULTS_DATA_DIR / 'sdi_data_fpb_backup2.csv'
        if backup.exists():
            sdi_path = backup
    sdi = pd.read_csv(sdi_path)
    print(f"Loaded {len(sdi)} sentences from {sdi_path}")

    # Collect every label column we have (single agents + interaction outputs)
    label_cols = ['vader_label', 'finbert_label', 'llm_label']
    short_names = {'vader_label': 'VADER', 'finbert_label': 'FinBERT',
                    'llm_label': 'Qwen-7B'}

    # Pull interaction labels into the single dataframe by sentence_id merge
    if 'sentence_id' in sdi.columns:
        for path in sorted(RESULTS_DATA_DIR.glob('interaction_results_*.csv')):
            stem = path.stem.replace('interaction_results_', '')
            try:
                inter = pd.read_csv(path)
            except Exception:
                continue
            if 'sentence_id' not in inter.columns:
                continue
            for col_name in ['vote_label', 'critic_label', 'debate_label']:
                if col_name not in inter.columns:
                    continue
                tag = col_name.replace('_label', '')
                key = f'{tag}@{stem}_label'
                if key not in sdi.columns:
                    sdi = sdi.merge(inter[['sentence_id', col_name]].rename(
                        columns={col_name: key}), on='sentence_id', how='left')
                    label_cols.append(key)
                    short_names[key] = f'{tag}@{stem}'

    print(f"Label columns: {[short_names[c] for c in label_cols]}")

    # Pairwise error-set Jaccard
    J = error_set_overlap(sdi, label_cols, gold_col='label_text')
    J_named = J.copy()
    J_named.index = [short_names[c] for c in label_cols]
    J_named.columns = [short_names[c] for c in label_cols]
    J_named.to_csv(RESULTS_DATA_DIR / 'e3_bias_overlap.csv')
    print("\nPairwise error-set Jaccard (lower = more orthogonal failures):")
    print(J_named.round(3).to_string())

    # Per-class F1
    f1_rows = []
    for c in label_cols:
        per_class_f1 = f1_score(sdi['label_text'], sdi[c], labels=CLASSES, average=None,
                                zero_division=0)
        per_class_recall = recall_score(sdi['label_text'], sdi[c], labels=CLASSES,
                                        average=None, zero_division=0)
        f1_rows.append({
            'agent': short_names[c],
            **{f'f1_{cls}': float(per_class_f1[i]) for i, cls in enumerate(CLASSES)},
            **{f'recall_{cls}': float(per_class_recall[i]) for i, cls in enumerate(CLASSES)},
            'f1_macro': float(per_class_f1.mean()),
        })
    pcdf = pd.DataFrame(f1_rows)
    pcdf.to_csv(RESULTS_DATA_DIR / 'e3_per_class.csv', index=False)
    print("\nPer-class F1:")
    print(pcdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Minority-class recall lift: best single agent vs best committee/protocol
    best_single_recall = pcdf[pcdf['agent'].isin(['VADER', 'FinBERT', 'Qwen-7B'])]['recall_negative'].max()
    best_committee_recall = pcdf['recall_negative'].max()
    lift = best_committee_recall - best_single_recall
    print(f"\nBest single-agent neg-class recall: {best_single_recall:.3f}")
    print(f"Best committee/protocol neg-class recall: {best_committee_recall:.3f}")
    print(f"Bias-cancellation lift on minority (negative) class recall: {lift:+.3f}")
    pd.DataFrame([{
        'best_single_recall_negative':    best_single_recall,
        'best_committee_recall_negative': best_committee_recall,
        'lift':                            lift,
    }]).to_csv(RESULTS_DATA_DIR / 'e3_minority_class_fnr.csv', index=False)

    # Visualisation: Jaccard heatmap
    fig, ax = plt.subplots(figsize=(0.5 + 0.55 * len(label_cols), 0.5 + 0.55 * len(label_cols)))
    sns.heatmap(J_named, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=1,
                cbar_kws={'label': 'error-set Jaccard'}, square=True, ax=ax)
    ax.set_title('Pairwise error-set overlap\n(lower = orthogonal failures)')
    fig.tight_layout()
    out = FIGURES_DIR / 'fig_e3_bias_overlap.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ saved {out}")


if __name__ == '__main__':
    main()
