# TriAgent Experimental Pipeline (English logical order)

This document traces the experimental pipeline end-to-end so that any
co-author can re-run, extend, or modify a single piece without
re-running the whole project. Steps are written in the **dependency
order** (each step uses outputs from earlier steps). Pure parallel
steps (no GPU contention) are flagged ⚡; sequential GPU steps are
flagged 🐌.

---

## Step 0 — Environment

* Hardware: 1 × NVIDIA RTX A5000 (24 GB), Python 3.10, PyTorch CUDA
  12.4, HuggingFace Transformers ≥ 4.45.
* Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
* Open-source dependencies: `vaderSentiment` (L1), `ProsusAI/finbert`
  (L2), `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct`,
  `Qwen/Qwen2.5-14B-Instruct` in 4-bit via `bitsandbytes`,
  `mistralai/Mistral-7B-Instruct-v0.3`,
  `yiyanghkust/finbert-tone-chinese`,
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Step 1 — L1: build the canonical committee on FPB 🐌

```bash
python experiments/L1_data_collection.py --dataset fpb --yes
```

Produces `results/data/committee_data.csv` with one row per FPB
sentence and one column block per agent (VADER / FinBERT /
Qwen-7B). This is the **foundation file** every later step reads.

## Step 2 — L1.5: multi-size LLM sweep 🐌 (sequential per size)

```bash
python experiments/L1p5_size_sweep.py --sizes 0.5B
python experiments/L1p5_size_sweep.py --sizes 1.5B
python experiments/L1p5_size_sweep.py --sizes 3B
python experiments/L1p5_size_sweep.py --sizes 14B    # 4-bit
python experiments/L1p5_size_sweep.py --sizes Mistral-7B
```

Appends `llm_{0p5b, 1p5b, 3b, 14b, mistral7b}_*` columns to
`committee_data.csv`. ⚡ Each size is independent — could run in
parallel on multi-GPU; we run sequentially on a single 24 GB card.

## Step 3 — L2: SDI metric + four-quadrant ⚡ CPU only

```bash
python experiments/L2_sdi_analysis.py
```

Adds `sdi_le, sdi_lr, sdi_er, sdi_max, sdi_mean, quadrant` columns
to `committee_data.csv`, saves the augmented dataframe as
`sdi_data.csv`, writes `fig_sdi_three_way.{pdf,png}`,
`fig_bias_diversity.{pdf,png}`, and `quadrant_summary.tex`.

## Step 4 — L2.5: interaction protocols 🐌

```bash
# vote (no extra LLM call)
python experiments/L2p5_interaction.py --protocols vote
# critic at each Qwen size
python experiments/L2p5_interaction.py --protocols critic --llm-size 1.5B --critic-trigger-col sdi_le --critic-threshold 0.4
python experiments/L2p5_interaction.py --protocols critic --llm-size 3B   --critic-trigger-col sdi_le --critic-threshold 0.4
python experiments/L2p5_interaction.py --protocols critic --llm-size 7B   --critic-trigger-col sdi_le --critic-threshold 0.4
# debate sweep across sizes
python experiments/L2p5_interaction.py --protocols debate --llm-size 1.5B --debate-trigger-col sdi_max --debate-threshold 0.5
python experiments/L2p5_interaction.py --protocols debate --llm-size 3B   --debate-trigger-col sdi_max --debate-threshold 0.5
python experiments/L2p5_interaction.py --protocols debate --llm-size 7B   --debate-trigger-col sdi_max --debate-threshold 0.5
# cross-family Mistral
python experiments/L2p5_interaction.py --protocols critic --llm-size Mistral-7B
python experiments/L2p5_interaction.py --protocols debate --llm-size Mistral-7B
```

Each run writes `interaction_results_<protocol>_<size>.csv` (per-row
predictions) and `interaction_summary_<protocol>_<size>.csv`
(F1 / cost summary).

## Step 5 — L3: cost-Pareto routing ⚡ CPU only

```bash
python experiments/L3_routing_pareto.py
```

Sweeps S0–S6 across escalation percentages, writes
`pareto_points_7b.csv`, `fig_main_pareto.pdf`,
`operating_points_7b.tex`.

## Step 6 — L3.5: cross-size scaling figures ⚡ CPU only

```bash
python experiments/L3p5_scaling.py
```

Generates `fig_scaling_inflection.pdf`,
`fig_interaction_vs_size.pdf`, `fig_pareto_multi_size.pdf` from the
combined per-size + per-protocol summaries.

## Step 7 — L4: edge predictor + three-granularity ablation ⚡

```bash
python experiments/L4_predictor.py --with-reasoning
```

Mines n-gram triggers, extracts ~100 features (word + phrase +
sentence-BERT PCA-16), trains LR and XGBoost, produces
`predictor_results.csv`, `fig_predictor_auc.pdf`,
`fig_feature_importance.pdf`, `predictor_ablation.tex`.

## Step 8 — L5: 20-ticker backtest 🐌→⚡ (yfinance HTTP + CPU)

```bash
python experiments/L5_backtest.py
```

Downloads OHLC for 20 tickers (one-time HTTP), runs the trade engine
across strategies, writes `backtest_results.csv`,
`backtest_summary_aggregate.csv`, `backtest_summary.tex`,
`fig_equity_curves.pdf`.

## Step 9 — L5.5: security experiments ⚡

```bash
python experiments/L5p5_e2_hallucination_detector.py
python experiments/L5p5_e3_bias_diversity_detail.py
python experiments/L5p5_e1_adversarial.py --generate    # CPU: builds perturbed sentences
python experiments/L5p5_e1_adversarial.py --run-committee   # 🐌 GPU: V+F+Qwen on perturbed
python experiments/L5p5_e1_adversarial.py --analyse    # CPU
```

Each writes its own `e{1,2,3}_*.csv` and a corresponding figure.

## Step 10 — L7: cross-lingual pilot 🐌→⚡

```bash
python experiments/L7_chinese_pilot.py --translate --n 1500
python experiments/L7_chinese_pilot.py --sweep --sizes 0.5B,1.5B,3B,7B
python experiments/L7p5_cross_lingual_committee.py
```

Translates 1500 FPB sentences to Chinese, runs the multi-size sweep
on the translation, then computes EN/ZH committee consistency and
the SCD cross-lingual canonicalizer numbers.

## Step 11 — L8: Shared Consensus Dictionary ⚡

```bash
python experiments/L8_public_dictionary.py
```

Builds the SCD on a 70/30 build/query split of FPB, sweeps the
similarity threshold τ, writes `scd_threshold_sweep.csv` and
`fig_scd_tradeoff.pdf`.

## Step 12 — L9: same-size persona vote (negative-control) 🐌

```bash
python experiments/L9_same_size_multiagent.py
```

Three Qwen-1.5B personas (bull / bear / neutral) majority-vote.
Confirms persona vote does not reproduce the critic plateau.

## Step 13 — paper rendering ⚡ CPU only

```bash
for s in paper/figures/code/make_*.py; do python "$s"; done
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Each `make_fig_*.py` reads its source CSV in `results/data/` and
writes both `.pdf` and `.png` to `paper/figures/`.

---

## What is parallelisable in this pipeline

* **Step 2** sizes are mutually independent — run on separate GPUs.
* **Step 4** protocol × size cells are independent — same.
* **Step 7, 9, 11** are pure CPU and run in seconds; parallel pointless.
* **Step 13** figure generation is embarrassingly parallel.

What is **NOT** parallelisable on a single 24 GB GPU: any two LLM
inference jobs that don’t fit in VRAM simultaneously. In practice
this means we run one Qwen size at a time.

## Re-running the paper from scratch (minimal recipe)

```bash
python experiments/L1_data_collection.py --dataset fpb --yes
for SIZE in 0.5B 1.5B 3B 14B Mistral-7B; do
    python experiments/L1p5_size_sweep.py --sizes "$SIZE"
done
python experiments/L2_sdi_analysis.py
for SIZE in 1.5B 3B 7B Mistral-7B; do
    python experiments/L2p5_interaction.py --protocols critic --llm-size "$SIZE"
    python experiments/L2p5_interaction.py --protocols debate --llm-size "$SIZE"
done
python experiments/L3_routing_pareto.py
python experiments/L3p5_scaling.py
python experiments/L4_predictor.py --with-reasoning
python experiments/L5_backtest.py
python experiments/L5p5_e2_hallucination_detector.py
python experiments/L5p5_e3_bias_diversity_detail.py
python experiments/L5p5_e1_adversarial.py --generate
python experiments/L5p5_e1_adversarial.py --run-committee
python experiments/L5p5_e1_adversarial.py --analyse
python experiments/L7_chinese_pilot.py --translate --n 1500
python experiments/L7_chinese_pilot.py --sweep --sizes 0.5B,1.5B,3B,7B
python experiments/L7p5_cross_lingual_committee.py
python experiments/L8_public_dictionary.py
python experiments/L9_same_size_multiagent.py
for s in paper/figures/code/make_*.py; do python "$s"; done
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

End-to-end wall-clock on a single A5000: ~6–8 hours
(LLM inference dominates; the 14 B 4-bit pass alone is ~1.5 hours).
