# Layer L4: Edge-Side Divergence Predictor (思路版)

## Goal

Train a lightweight classifier that predicts whether a sentence will have
high SDI — using ONLY features computable at the VADER (edge) stage.

This makes the routing **deployable**: no need to wait for FinBERT/LLM to
compute SDI; predict it upfront.

## Why This Layer Matters

L3 shows "SDI routing is good *if* you know the SDI." But in production, you 
don't know the SDI until you've called all three agents — defeating the 
purpose. L4 closes this loop by showing SDI itself is predictable from edge 
features.

This is what turns the paper from "post-hoc analysis" into "deployable system."

## Why Bigram/Trigram Features Matter — KEY INNOVATION

Single-word triggers (like "loss") are ambiguous:
- "loss narrowed" → positive
- "loss widened" → negative

**Bigram/trigram features capture this context.** This is a clean ablation 
contribution: show that adding bigram features bumps AUC meaningfully (e.g., 
0.78 → 0.85).

This was missing from the original code (which only did unigrams). It's also 
something Isabel pointed out independently — "组合词" — making it a natural 
contribution to highlight.

## Inputs

`results/data/sdi_data.csv` (from L2)

## What to Implement

### `src/mining/log_odds.py`

Improve the original log-odds analysis to support n-grams:

```python
def mine_ngram_triggers(df, ngram_range=(1, 3), min_df=3, ...):
    """
    Returns DataFrame with columns:
        ngram, n (1/2/3), high_sdi_count, low_sdi_count, log_odds, z_score
    Sorted by z_score descending.
    """
```

Use `CountVectorizer` with `ngram_range=(1, 3)` and `token_pattern=r'[a-zA-Z]{3,}'`.

Save the top ~50-100 triggers (split by n) to `data/lexicons/trigger_ngrams.json`.

### `src/predictor/features.py`

Extract ~25 features per sentence, all computable at the VADER stage:

**VADER outputs (free)**:
- `vader_score`, `vader_pos`, `vader_neg`, `vader_neu`, `vader_confidence`

**Sentence structure**:
- `word_count`, `char_count`
- `has_number`, `has_currency_symbol` (regex)
- `has_contrast_word` (but/however/while/although/yet)
- `has_negation` (not/no/n't/without)

**Unigram triggers** (binary or count, from mined lexicon):
- `has_trigger_loss`, `has_trigger_decreased`, `has_trigger_profit`, ...
- `unigram_trigger_count` (total)

**⭐ Bigram triggers** (KEY CONTRIBUTION):
- `has_bigram_loss_narrowed`, `has_bigram_profit_declined`, 
  `has_bigram_operating_profit`, `has_bigram_net_loss`, ...
- `bigram_trigger_count`

**Trigram triggers**:
- `trigram_trigger_count` (often sparse, count is enough)

### `src/predictor/train.py`

Trains five models for ablation comparison:

1. **Random** — predict 50/50
2. **LR — unigram only** — VADER + sentence stats + unigram triggers
3. **LR — uni + bigram** — adds bigram trigger features ⭐ main method
4. **LR — uni + bigram + trigram** — adds trigram features
5. **XGBoost — all features** — performance ceiling

Stratified 80/20 train/test split. Random seed = 42.

Target: `y = (df['sdi_max'] > 0.7).astype(int)` — binary high/low SDI.

### `src/predictor/evaluate.py`

For each model:
- AUC-ROC
- Precision @ 20% recall (if budget allows escalating top-20%, how many 
  high-SDI samples do we catch?)
- Feature importance (LR coefficients with sign; XGBoost gain)

### `experiments/L4_predictor.py`

Pipeline:
1. Load sdi_data.csv
2. Mine n-gram triggers via log_odds.py
3. Extract features
4. Train 5 models
5. Generate ablation table and ROC plot
6. Plug LR-uni+bigram predictions into L3's routing simulator → compare 
   "Predictor-routed" vs "Oracle-SDI-routed" on Pareto frontier

## Output Files

- `data/lexicons/trigger_ngrams.json`
- `results/data/predictor_results.csv` — per-model metrics
- `results/figures/fig_predictor_auc.png` — ROC curves for all 5 models
- `results/figures/fig_feature_importance.png` — top features bar chart
- `results/figures/fig_predictor_routing.png` — Pareto with predictor vs oracle
- `results/tables/predictor_ablation.tex` — main ablation table

## The Ablation Table (the L4 highlight)

| Model | AUC | P@20%R | Top Features |
|-------|-----|--------|--------------|
| Random | 0.50 | 0.20 | — |
| LR — unigram | ~0.78 | ~0.45 | trigger_loss, vader_conf, trigger_decreased |
| **LR — uni+bigram** | **~0.85** | **~0.62** | **bigram_loss_narrowed, bigram_profit_declined, ...** |
| LR — uni+bi+trigram | ~0.86 | ~0.64 | (mostly same as bigram) |
| XGBoost | ~0.88 | ~0.68 | (gain on numeric features) |

The story: bigrams add ~7 AUC points. Going further (trigrams, XGBoost) gives 
diminishing returns. **LR-uni+bigram is the sweet spot for deployment** 
(interpretable, fast, almost-best AUC).

## Routing-with-Predictor Comparison

Add to the Pareto plot from L3 a new curve:
- **"Predictor routing"**: use LR-uni+bigram to decide escalation, not oracle 
  SDI

Expected: predictor curve sits 2-5 F1 points below oracle curve, but is 
**fully deployable**. This is the closing argument.

## Checkpoint Criteria

Before moving to L5, verify:

1. **LR-unigram AUC ≥ 0.75**
2. **LR-uni+bigram AUC ≥ 0.80** ⭐ (proves bigrams help)
3. **Predictor routing reaches ≥ 85% of Oracle SDI routing's F1** at the 
   same cost
4. **Top features make sense**: should see things like `bigram_loss_narrowed`, 
   `bigram_profit_declined`, `vader_confidence`, etc. — not random words.

If LR-uni+bigram AUC < 0.78, troubleshoot:
- Is the bigram lexicon comprehensive? Mine more (lower min_df).
- Add LLM reasoning embedding (sentence-BERT) as ablation feature
- Try gradient boosting instead of LR

## Bonus: LLM Reasoning Embedding (optional ablation)

If time permits:

```python
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
reasoning_emb = embedder.encode(df['llm_reasoning'].tolist())  # ~384 dims

# Reduce dim with PCA → 16 dims
# Add as features
```

This is **not deployable** (requires running L3 first), but as an ablation 
"upper bound", it shows whether reasoning text contains additional signal. 
Mark this clearly as "non-deployable ablation" in the paper.

## Coding Notes

- Don't recompute SDI here — it's already in `sdi_data.csv`
- The trigger lexicon should be saved as JSON, not pickled (interpretability)
- Use `sklearn.linear_model.LogisticRegression(max_iter=1000, C=1.0)` — 
  default reasonable
- For XGBoost, use `xgboost.XGBClassifier(n_estimators=100, max_depth=4)` 
  — keep it simple

## What This Layer Feeds Into

- **Paper §4** — Divergence Mining and Edge-Side Prediction
- **Paper §5.4** — Predictor performance subsection
- **L5 backtest** — uses predictor (not oracle SDI) to gate escalation, 
  showing the deployable version actually trades well too
