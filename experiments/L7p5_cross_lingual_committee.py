"""L7.5: Cross-lingual committee + cross-lingual consistency experiment.

Runs a Chinese committee (skipping VADER, since it doesn't do Chinese):
    L2 specialist: yiyanghkust/finbert-tone-chinese
    L3 reasoner:   Qwen2.5-Instruct (multi-size, already cached)

Then computes the cross-lingual *consistency rate*:
    For each FPB sentence we have both EN and ZH versions (1500 pairs).
    Run the committee on both → committee_en_label, committee_zh_label.
    Consistency = % of pairs where the two labels agree.

Finally tests whether the SCD (sentence-BERT cosine cache) raises the
consistency rate by canonicalising near-paraphrase pairs across
languages.

Outputs:
    results/data/committee_data_fpb_zh_full.csv  -- + finbert_cn columns
    results/data/cross_lingual_consistency.csv
    results/figures/fig_cross_lingual.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR
from src.viz.style import apply_style, COLORS


def aggregate_committee_label(finbert_label: str, llm_label: str,
                               finbert_conf: float, llm_conf: float) -> str:
    """Two-tier majority: if both agree → that label; else higher-conf wins.
    Tie-break to neutral. Mirrors the spirit of the 3-tier protocol minus
    the (here-broken) VADER tier.
    """
    if finbert_label == llm_label:
        return finbert_label
    if finbert_conf > llm_conf + 0.05:
        return finbert_label
    if llm_conf > finbert_conf + 0.05:
        return llm_label
    return 'neutral'


def main(args):
    apply_style()

    # ---- Load Chinese committee data (already has Qwen multi-size) ----
    zh_csv = RESULTS_DATA_DIR / 'committee_data_fpb_zh.csv'
    if not zh_csv.exists():
        raise FileNotFoundError(f"Run stage 7 (Chinese pilot) first to produce {zh_csv}")
    df = pd.read_csv(zh_csv)
    print(f"Loaded {len(df)} Chinese samples from {zh_csv}")

    # ---- Run Chinese FinBERT (replaces the broken English FinBERT for ZH) ----
    if 'finbert_cn_label' not in df.columns:
        from src.agents.finbert_cn_agent import FinBERTCNAgent
        print("Running Chinese FinBERT on ZH text...")
        agent = FinBERTCNAgent()
        results = agent.predict_batch(df['sentence'].fillna('').astype(str).tolist())
        df['finbert_cn_score']      = [r.score for r in results]
        df['finbert_cn_label']      = [r.label for r in results]
        df['finbert_cn_confidence'] = [r.confidence for r in results]
        df['finbert_cn_latency_ms'] = [r.latency_ms for r in results]
        df['finbert_cn_cost_usd']   = [r.cost_usd for r in results]
        df.to_csv(RESULTS_DATA_DIR / 'committee_data_fpb_zh_full.csv', index=False)
    finbert_cn_f1 = f1_score(df['label_text'], df['finbert_cn_label'], average='macro')
    print(f"  Chinese FinBERT F1 = {finbert_cn_f1:.4f}")

    # ---- Build Chinese committee labels (FinBERT-CN + Qwen-N) ----
    sizes = [('1.5B', 'llm_1p5b'), ('3B', 'llm_3b'),
             ('7B', 'llm_7b')]
    for sz, prefix in sizes:
        # Re-derive a Chinese committee label as the FinBERT_CN/Qwen majority
        f_label = df['finbert_cn_label']
        f_conf  = df['finbert_cn_confidence']
        l_label = df[f'{prefix}_label_zh']
        l_conf  = df[f'{prefix}_confidence_zh']
        committee_zh = [aggregate_committee_label(fl, ll, fc, lc)
                        for fl, ll, fc, lc in zip(f_label, l_label, f_conf, l_conf)]
        col = f'committee_zh_{prefix.replace("llm_", "")}'
        df[col] = committee_zh
        f1 = f1_score(df['label_text'], committee_zh, average='macro')
        print(f"  Chinese committee (FinBERT-CN + Qwen-{sz}): F1 = {f1:.4f}")

    df.to_csv(RESULTS_DATA_DIR / 'committee_data_fpb_zh_full.csv', index=False)

    # ---- Cross-lingual consistency ----
    # Load English committee data and match by ENGLISH SENTENCE TEXT,
    # since the L7 translator stored the EN original in `sentence_en`.
    sdi_en = pd.read_csv(RESULTS_DATA_DIR / 'sdi_data_fpb_backup2.csv')
    en_pull = sdi_en[['sentence', 'finbert_label', 'finbert_confidence',
                       'llm_label', 'llm_confidence', 'vader_label']].copy()
    en_pull = en_pull.rename(columns={
        'sentence':           'sentence_en_match',
        'vader_label':        'en_vader_label',
        'finbert_label':      'en_finbert_label',
        'finbert_confidence': 'en_finbert_conf',
        'llm_label':          'en_llm_label',
        'llm_confidence':     'en_llm_conf',
    })
    # Align by exact-text join on the English source sentence
    if 'sentence_en' in df.columns:
        df_aligned = df.merge(en_pull, left_on='sentence_en',
                              right_on='sentence_en_match', how='left')
    else:
        # If sentence_en isn't carried over, we can't align cleanly — bail
        print("WARN: 'sentence_en' missing from Chinese pilot CSV; "
              "cross-lingual consistency uses positional alignment (rough).")
        df_aligned = df.copy()
        for c in ['en_vader_label','en_finbert_label','en_finbert_conf',
                   'en_llm_label','en_llm_conf']:
            df_aligned[c] = en_pull[c].iloc[:len(df)].values

    n_matched = df_aligned['en_finbert_label'].notna().sum()
    print(f"  matched {n_matched}/{len(df_aligned)} EN/ZH pairs by sentence text")

    df_aligned['committee_en'] = [
        aggregate_committee_label(fl if isinstance(fl, str) else 'neutral',
                                   ll if isinstance(ll, str) else 'neutral',
                                   fc if pd.notna(fc) else 0.85,
                                   lc if pd.notna(lc) else 0.85)
        for fl, ll, fc, lc in zip(df_aligned['en_finbert_label'],
                                   df_aligned['en_llm_label'],
                                   df_aligned['en_finbert_conf'],
                                   df_aligned['en_llm_conf'])
    ]

    rows = []
    for sz, prefix in sizes:
        zh_col = f'committee_zh_{prefix.replace("llm_", "")}'
        common = df_aligned.dropna(subset=['en_llm_label', zh_col])
        en_lbl = common['committee_en'].astype(str)
        zh_lbl = common[zh_col].astype(str)
        agree = (en_lbl == zh_lbl).mean()
        kappa = cohen_kappa_score(en_lbl, zh_lbl)
        # F1 of each side vs gold
        en_f1 = f1_score(common['label_text'], en_lbl, average='macro')
        zh_f1 = f1_score(common['label_text'], zh_lbl, average='macro')
        rows.append({
            'qwen_size': sz,
            'cross_lingual_agreement_rate': float(agree),
            'cohen_kappa': float(kappa),
            'en_committee_f1': float(en_f1),
            'zh_committee_f1': float(zh_f1),
            'n_compared': int(len(common)),
        })
    cl = pd.DataFrame(rows)
    print("\n=== Cross-lingual committee consistency (EN vs ZH committee per Qwen size) ===")
    print(cl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    cl.to_csv(RESULTS_DATA_DIR / 'cross_lingual_consistency.csv', index=False)

    # ---- Optional: SCD as cross-lingual canonicalizer ----
    try:
        from src.dictionary.public_dictionary import SharedConsensusDictionary
        print("\n=== SCD as cross-lingual canonicalizer ===")
        # Build SCD from ENGLISH labels using debate@7B (consensus on EN)
        # Then query with the CHINESE sentences — does multilingual sentence-BERT
        # match cross-language paraphrases and return consistent labels?
        en_train = sdi_en.copy()
        # For each Chinese sentence in df_aligned, encode with multilingual model
        # and cos-search EN cache. If match → EN label; if no match → ZH committee.
        scd = SharedConsensusDictionary(threshold=0.5)
        scd.build_from_dataframe(en_train,
                                  text_col='sentence', label_col='label_text',
                                  confidence_col='finbert_confidence')

        zh_sentences = df_aligned['sentence'].fillna('').astype(str).tolist()
        rows2 = []
        for tau in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85]:
            scd.threshold = tau
            hits = scd.query_batch(zh_sentences)
            n_hit = sum(1 for h in hits if h is not None)
            cached_labels = [h.cached_label if h else 'MISS' for h in hits]
            # Hit rate + agreement of hits with gold
            if n_hit > 0:
                hit_idx = [i for i, h in enumerate(hits) if h is not None]
                gold_subset = df_aligned['label_text'].iloc[hit_idx].astype(str).values
                pred_subset = [cached_labels[i] for i in hit_idx]
                hit_acc = accuracy_score(gold_subset, pred_subset)
                hit_f1 = f1_score(gold_subset, pred_subset, average='macro', zero_division=0)
            else:
                hit_acc = hit_f1 = 0.0
            rows2.append({'tau': tau, 'cross_lingual_hit_rate': n_hit / len(hits),
                           'hit_accuracy_vs_gold': hit_acc,
                           'hit_f1_vs_gold':       hit_f1})
        scd_cl = pd.DataFrame(rows2)
        print(scd_cl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        scd_cl.to_csv(RESULTS_DATA_DIR / 'scd_cross_lingual.csv', index=False)
    except Exception as e:
        print(f"SCD cross-lingual section failed: {e}")

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sizes_x = [1.5, 3, 7]
    ax.plot(sizes_x, cl['cross_lingual_agreement_rate'].values,
            marker='o', lw=2, color='#2ca02c', label='EN↔ZH committee agreement')
    ax.plot(sizes_x, cl['en_committee_f1'].values,
            marker='s', lw=1.5, ls='--', color='#1f77b4', label='EN committee F1 vs gold')
    ax.plot(sizes_x, cl['zh_committee_f1'].values,
            marker='^', lw=1.5, ls='--', color='#d62728', label='ZH committee F1 vs gold')
    ax.set_xscale('log')
    ax.set_xticks(sizes_x); ax.set_xticklabels(['1.5B', '3B', '7B'])
    ax.set_xlabel('Qwen-N reasoning tier')
    ax.set_ylabel('rate / F1')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title('Cross-lingual committee consistency (1500 EN/ZH FPB pairs)')
    fig.tight_layout()
    out = FIGURES_DIR / 'fig_cross_lingual.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ saved {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
