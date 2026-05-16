"""L8: Shared Consensus Dictionary (SCD) — embedding-cached committee decisions.

Pipeline:
  1. Split sdi_data.csv into 70% build / 30% query
  2. Build SCD entries from the build set (use debate@7B labels as the
     "consensus" — falling back to FinBERT if debate not available)
  3. For τ ∈ {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}, query each test sample:
       * Hit  → use cached label
       * Miss → use the live committee label (here, debate@7B again)
  4. Report per-τ:
       * hit rate
       * F1-Macro (overall, hit-only, miss-only)
       * cost saving = (hit_rate * always_l3_cost) per query
       * latency saving (always_l3_latency * hit_rate)

Outputs:
  results/data/scd_threshold_sweep.csv
  results/data/scd_examples.csv      — random matched-pairs for paper
  results/figures/fig_scd_tradeoff.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR, TABLES_DIR
from src.dictionary.public_dictionary import SharedConsensusDictionary
from src.viz.style import apply_style, COLORS


def main(args):
    apply_style()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    sdi_path = Path(args.input) if args.input else RESULTS_DATA_DIR / 'sdi_data_fpb_backup2.csv'
    if not sdi_path.exists():
        sdi_path = RESULTS_DATA_DIR / 'sdi_data.csv'
    sdi = pd.read_csv(sdi_path)
    print(f"Loaded {len(sdi)} sentences from {sdi_path}")

    # Pull the 'consensus' label preference: debate@7B > critic@1.5B > FinBERT > gold
    # (we treat the strongest-validated committee output as the consensus to cache)
    debate_path = RESULTS_DATA_DIR / 'interaction_results_7b.csv'
    consensus = None
    if debate_path.exists():
        d = pd.read_csv(debate_path)
        if 'debate_label' in d.columns and 'sentence_id' in d.columns:
            sdi = sdi.merge(d[['sentence_id', 'debate_label']],
                            on='sentence_id', how='left')
            consensus = 'debate_label'
            print(f"  Using debate@7B as committee consensus label")
    if consensus is None:
        consensus = 'finbert_label'
        print(f"  Falling back to FinBERT as consensus label")
    sdi['__consensus__'] = sdi[consensus].fillna(sdi['finbert_label'])

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------
    train, test = train_test_split(sdi, test_size=args.test_frac,
                                    random_state=42, stratify=sdi['label_text'])
    print(f"  build_set = {len(train)},  query_set = {len(test)}")

    # ------------------------------------------------------------------
    # Build SCD from train set (live LLM cost we'd save: llm_cost_usd)
    # ------------------------------------------------------------------
    scd = SharedConsensusDictionary(threshold=0.5)
    scd.build_from_dataframe(
        train,
        text_col='sentence', label_col='__consensus__',
        confidence_col='finbert_confidence',
    )

    # ------------------------------------------------------------------
    # Sweep τ
    # ------------------------------------------------------------------
    print("\nSweeping similarity threshold...")
    test_sentences = test['sentence'].tolist()
    test_gold      = test['label_text'].to_numpy()
    live_label     = test['__consensus__'].to_numpy()           # what committee would say
    live_cost      = test['llm_cost_usd'].sum() if 'llm_cost_usd' in test.columns else 0.0
    live_lat       = test['llm_latency_ms'].sum() if 'llm_latency_ms' in test.columns else 0.0

    # Precompute hits at the LOOSEST threshold (then filter)
    scd.threshold = 0.0
    raw_hits = scd.query_batch(test_sentences, batch_size=64)
    hit_sims  = np.array([h.similarity if h else -1.0 for h in raw_hits])
    hit_label = np.array([h.cached_label if h else None for h in raw_hits])

    rows = []
    for tau in args.thresholds:
        is_hit = hit_sims >= tau
        n_hit  = int(is_hit.sum())
        # Final label: cache when hit, else live committee
        final = np.where(is_hit, hit_label, live_label)

        # F1 partitions
        f1_overall = f1_score(test_gold, final, average='macro', zero_division=0)
        f1_hit_only = (f1_score(test_gold[is_hit], final[is_hit],
                                average='macro', zero_division=0)
                       if n_hit else 0.0)
        f1_miss_only = (f1_score(test_gold[~is_hit], final[~is_hit],
                                  average='macro', zero_division=0)
                        if (~is_hit).sum() else 0.0)
        f1_committee_only = f1_score(test_gold, live_label,
                                      average='macro', zero_division=0)

        # Cost / latency saving — assumes live committee uses the LLM
        # for every miss; on hit we skip the LLM entirely
        per_query_llm_cost = (live_cost / max(len(test), 1))
        per_query_llm_lat  = (live_lat  / max(len(test), 1))
        savings_dollars    = per_query_llm_cost * n_hit
        savings_ms         = per_query_llm_lat  * n_hit

        rows.append({
            'tau':            tau,
            'hit_rate':       float(is_hit.mean()),
            'n_hit':          n_hit,
            'n_miss':         int((~is_hit).sum()),
            'f1_overall':     float(f1_overall),
            'f1_hit_only':    float(f1_hit_only),
            'f1_miss_only':   float(f1_miss_only),
            'f1_committee_only': float(f1_committee_only),
            'savings_$_per_query':  float(per_query_llm_cost) if n_hit else 0.0,
            'savings_ms_per_query': float(per_query_llm_lat)  if n_hit else 0.0,
            'total_savings_$':      float(savings_dollars),
            'total_savings_ms':     float(savings_ms),
        })
    sweep = pd.DataFrame(rows)
    print("\n=== τ sweep ===")
    print(sweep[['tau','hit_rate','f1_overall','f1_hit_only','f1_miss_only',
                 'f1_committee_only','total_savings_$']]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sweep.to_csv(RESULTS_DATA_DIR / 'scd_threshold_sweep.csv', index=False)

    # ------------------------------------------------------------------
    # Sample of matched pairs for the paper appendix
    # ------------------------------------------------------------------
    scd.threshold = 0.85   # paper-quoted threshold
    typical_hits = scd.query_batch(test_sentences[:200], batch_size=64)
    examples = []
    for i, h in enumerate(typical_hits[:50]):
        if h is None or h.similarity < 0.85:
            continue
        examples.append({
            'query_sentence':  test_sentences[i],
            'matched_in_dict': h.matched_sentence,
            'cached_label':    h.cached_label,
            'gold_label':      test['label_text'].iloc[i],
            'similarity':      h.similarity,
        })
    pd.DataFrame(examples).to_csv(RESULTS_DATA_DIR / 'scd_examples.csv', index=False)
    print(f"  ✓ saved {len(examples)} matched-pair examples to scd_examples.csv")

    # ------------------------------------------------------------------
    # Figure: hit-rate vs F1 vs cost-saving trade-off
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7, 4.4))
    ax1.set_xlabel('Similarity threshold τ')
    ax1.set_ylabel('F1-Macro on test set')
    ax1.plot(sweep['tau'], sweep['f1_overall'],
             marker='o', lw=2.0, color='#1f77b4', label='SCD-hybrid F1 (cache + miss→committee)')
    ax1.axhline(sweep['f1_committee_only'].iloc[0], ls='--', color='#888888',
                 label=f"Always-committee F1 = {sweep['f1_committee_only'].iloc[0]:.3f}")
    ax1.set_ylim(0.6, 0.95)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cache hit rate')
    ax2.plot(sweep['tau'], sweep['hit_rate'],
             marker='s', lw=2.0, color='#d62728', label='Hit rate')
    ax2.set_ylim(0, 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=8)
    ax1.set_title('Shared Consensus Dictionary — accuracy / hit-rate trade-off')
    fig.tight_layout()
    out = FIGURES_DIR / 'fig_scd_tradeoff.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ saved {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None,
                        help='sdi_data.csv path (default: FPB backup)')
    parser.add_argument('--test-frac', type=float, default=0.30)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    args = parser.parse_args()
    main(args)
