"""TFNS phrase-level evidence pipeline.

Validates the "customer-language → phrase-level features matter more"
claim by running the L4 predictor on Twitter Financial News Sentiment
(short, fragmentary financial text) and comparing the bigram lift to
what we measured on FPB.

Prereq: run `python experiments/L1_data_collection.py --dataset tfns
--skip-llm --yes` first to populate committee_data_tfns.csv with
VADER + FinBERT predictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR, TABLES_DIR, LEXICONS_DIR
from src.mining.log_odds import mine_ngram_triggers, save_top_triggers
from src.predictor.features import (
    extract_features, load_trigger_lexicon, feature_columns_for,
)
from src.predictor.train import (
    train_split, train_random, train_lr, train_xgb, lr_top_features,
)
from src.viz.style import apply_style


def fig_predictor_auc(models, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for m in models:
        if len(np.unique(m.test_y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(m.test_y, m.test_pred_proba)
        ax.plot(fpr, tpr, lw=1.6,
                label=f"{m.name}  (AUC={m.metrics['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='#888', lw=0.8)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('TFNS edge predictor — ROC')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


def main(args):
    apply_style()

    src_csv = RESULTS_DATA_DIR / 'committee_data_tfns.csv'
    if not src_csv.exists():
        raise FileNotFoundError(
            f"Run `python experiments/L1_data_collection.py --dataset tfns "
            f"--skip-llm --yes` first to produce {src_csv}")
    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} TFNS rows from {src_csv}")
    print(f"Label dist: {df['label_text'].value_counts().to_dict()}")

    # Compute SDI_LE (lexicon vs expert) — the only SDI possible without LLM
    df['sdi_le'] = (df['vader_score'] - df['finbert_score']).abs()
    print(f"SDI_LE summary: mean={df['sdi_le'].mean():.3f}, "
          f">0.7 = {(df['sdi_le']>0.7).mean():.1%}, "
          f">{args.threshold} = {(df['sdi_le']>args.threshold).mean():.1%}")

    # Mine n-gram triggers from the high-SDI subset
    print(f"\nMining n-gram triggers (target=sdi_le>{args.threshold})...")
    triggers = mine_ngram_triggers(
        df, target_col='sdi_le', target_threshold=args.threshold,
        ngram_range=(1, 3), min_df=args.min_df,
    )
    print(f"  total candidates: {len(triggers)}")
    print(f"  by n: {triggers['n'].value_counts().to_dict()}")
    print("\n  Top 6 unigram triggers:")
    print(triggers[triggers['n'] == 1].head(6)[['ngram', 'z_score']].to_string(index=False))
    print("\n  Top 6 bigram triggers:")
    print(triggers[triggers['n'] == 2].head(6)[['ngram', 'z_score']].to_string(index=False))
    print("\n  Top 6 trigram triggers:")
    print(triggers[triggers['n'] == 3].head(6)[['ngram', 'z_score']].to_string(index=False))

    lex_path = LEXICONS_DIR / 'trigger_ngrams_tfns.json'
    save_top_triggers(triggers, out_path=lex_path, top_per_n=args.top_per_n)
    print(f"  ✓ saved trigger lexicon → {lex_path}")
    lexicon = load_trigger_lexicon(lex_path)

    # Extract features
    X = extract_features(df, lexicon)
    y = (df['sdi_le'] > args.threshold).astype(int).to_numpy()
    print(f"\nFeature matrix: {X.shape}, positive rate = {y.mean():.3f}")

    X_tr, X_te, y_tr, y_te = train_split(X, y, test_size=0.2, seed=42)
    print(f"  train={len(X_tr)}, test={len(X_te)}")

    # Five-model ablation, same as FPB L4
    print("\nTraining models...")
    models = []
    models.append(train_random(y_tr, y_te))
    models.append(train_lr(X_tr, y_tr, X_te, y_te,
                           name='LR-unigram',
                           feature_cols=feature_columns_for('unigram', X)))
    models.append(train_lr(X_tr, y_tr, X_te, y_te,
                           name='LR-uni+bigram',
                           feature_cols=feature_columns_for('unibi', X)))
    models.append(train_lr(X_tr, y_tr, X_te, y_te,
                           name='LR-uni+bi+trigram',
                           feature_cols=feature_columns_for('unibitri', X)))
    models.append(train_xgb(X_tr, y_tr, X_te, y_te,
                            feature_cols=feature_columns_for('unibitri', X)))

    # Ablation table
    rows = [{'model': m.name, **m.metrics} for m in models]
    df_summary = pd.DataFrame(rows)
    print("\n=== TFNS ablation summary ===")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df_summary.to_csv(RESULTS_DATA_DIR / 'predictor_results_tfns.csv', index=False)
    fig_predictor_auc(models, FIGURES_DIR / 'fig_predictor_auc_tfns.png')

    # LR-uni+bigram top features for paper
    main_lr = next(m for m in models if m.name == 'LR-uni+bigram')
    top = lr_top_features(main_lr, k=12)
    print("\n  Top 12 LR-uni+bigram features (TFNS):")
    print(top.to_string(index=False))

    with open(TABLES_DIR / 'predictor_ablation_tfns.tex', 'w') as f:
        f.write("% Auto-generated by L4_tfns_phrase_evidence.py\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model & AUC-ROC & AUC-PR & P@R20 & P@top10 \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['auc_roc']:.3f} & {r['auc_pr']:.3f} & "
                    f"{r['p_at_r20']:.3f} & {r['p_at_top10']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"\n  ✓ saved predictor_ablation_tfns.tex")

    # Compare with FPB if available
    fpb_csv = RESULTS_DATA_DIR / 'predictor_results.csv'
    if fpb_csv.exists():
        fpb = pd.read_csv(fpb_csv)
        print("\n=== Bigram lift comparison: FPB vs TFNS ===")
        for cmp in [('LR-unigram', 'LR-uni+bigram'),
                    ('LR-unigram', 'LR-uni+bi+trigram')]:
            try:
                fpb_lift = (fpb.set_index('model').loc[cmp[1], 'auc_roc']
                            - fpb.set_index('model').loc[cmp[0], 'auc_roc'])
                tfns_lift = (df_summary.set_index('model').loc[cmp[1], 'auc_roc']
                             - df_summary.set_index('model').loc[cmp[0], 'auc_roc'])
                print(f"  {cmp[0]} -> {cmp[1]}:  "
                      f"FPB lift = {fpb_lift:+.4f},  "
                      f"TFNS lift = {tfns_lift:+.4f}")
            except Exception as e:
                print(f"  comparison failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='SDI_LE threshold for the high-SDI positive class.')
    parser.add_argument('--min-df', type=int, default=3)
    parser.add_argument('--top-per-n', type=int, default=50)
    args = parser.parse_args()
    main(args)
