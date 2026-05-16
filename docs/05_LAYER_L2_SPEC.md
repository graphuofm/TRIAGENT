# Layer L2: Three-Way SDI + Bias Diversity (思路版)

## Goal

Extend SDI from pairwise (vader-finbert) to three-way decomposition. Establish
"committee behavior profile" via four-quadrant classification and quantify
agent bias diversity.

This layer turns L1's raw committee predictions into the **structural insights** 
that motivate L3's routing strategies.

## Inputs

`results/data/committee_data.csv` (from L1)

## Conceptual Approach

### The Big Idea

A pair of agents that disagree (high SDI) tells you "something interesting is
happening here," but doesn't tell you *what*. With three agents, the *pattern* 
of disagreement becomes diagnostic:

- **All three agree** → easy case, use cheapest agent
- **L1 disagrees but L2/L3 agree** → domain shift, use specialist
- **L2/L3 disagree** → genuinely ambiguous, escalate to reasoner or human
- **All three disagree** → chaos zone, almost certainly hard

This is the conceptual jump from "SDI as diagnostic" to "SDI as orchestration 
signal." Make sure the figures and writing in this layer convey it clearly.

## What to Implement

### `src/metrics/sdi.py`

A function that takes the committee dataframe and adds:

- `sdi_le` — abs(vader_score - finbert_score) — Lexicon vs Expert
- `sdi_lr` — abs(vader_score - llm_score) — Lexicon vs Reasoner
- `sdi_er` — abs(finbert_score - llm_score) — Expert vs Reasoner
- `sdi_max`, `sdi_mean` — aggregate measures
- `quadrant` — categorical label per the four-quadrant scheme

Use `SDI_HIGH=0.7` and `SDI_LOW=0.3` from config. Make these tunable, not 
hardcoded.

### `src/metrics/bias_diversity.py`

Two metrics:

1. **Pairwise Cohen's kappa** on labels (not scores). Three values: 
   `kappa(vader, finbert)`, `kappa(vader, llm)`, `kappa(finbert, llm)`. Low 
   kappa = high diversity = committee makes sense.

2. **Per-sample disagreement entropy.** Take the three labels, compute the 
   distribution, return Shannon entropy. Aggregate to get a corpus-level 
   number.

### `experiments/L2_sdi_analysis.py`

Should produce:

1. Statistical analysis (replicate original ANOVA finding):
   - One-way ANOVA on SDI across gold sentiment classes
   - Pairwise Welch's t-tests with Cohen's d
   - Confirm `negative` class has highest mean SDI

2. Figures (use IJCAI-friendly style from `src/viz/style.py`):
   - **fig_sdi_three_way.png** — 3 panels: histograms of SDI_LE/LR/ER 
     overlaid, scatter of SDI_LE vs SDI_ER colored by quadrant, SDI 
     distribution by gold label
   - **fig_bias_diversity.png** — 2 panels: 3×3 kappa heatmap, 
     disagreement entropy distribution

3. Tables (LaTeX format, save to `results/tables/`):
   - **quadrant_summary.tex** — for each of 4 quadrants: count, %, 
     gold label distribution, agent accuracies, mean SDIs

## Quadrant Classification Logic

```
def classify_quadrant(row, hi=0.7, lo=0.3):
    le, er = row['sdi_le'], row['sdi_er']
    if le < lo and er < lo:    return 'consensus'      # all agree
    if le > hi and er < lo:    return 'domain_shift'   # only L1 wrong
    if er > hi:                return 'ambiguous'      # L2/L3 disagree
    return 'mixed'                                      # everything else
```

## Checkpoint Criteria

Before moving to L3, verify:

1. **Reproduction**: Negative-class mean SDI_LE ≈ 0.945. ANOVA F >> 1, 
   p < 0.001.

2. **Quadrant breakdown reasonable**:
   - Consensus: 40-55%
   - Domain shift: 15-25%
   - Ambiguous: 8-15%
   - Mixed: rest
   
   If consensus < 25% or > 70%, your thresholds are off — tune them.

3. **Quadrant-accuracy alignment**: VADER's accuracy in `consensus` should be 
   > 90%. VADER's accuracy in `domain_shift` should be < 50%. If these don't 
   hold, something's broken upstream.

4. **Kappa values make sense**:
   - kappa(vader, finbert) ≈ 0.3-0.5 (low → they disagree a lot)
   - kappa(finbert, llm) ≈ 0.7-0.85 (high → both are good)
   - kappa(vader, llm) somewhere in between

## What This Layer Feeds Into

- **L3 routing** uses `sdi_le` and `sdi_er` as routing signals directly
- **L4 predictor** uses `sdi_max > 0.7` as the prediction target
- **Paper §5.2** uses the bias diversity figures
- **Paper §3.2** references the four-quadrant interpretation as motivation

## Coding Notes

- Don't over-engineer this layer. It's mostly pandas + sklearn + matplotlib.
- The whole script should be < 200 LOC if done cleanly.
- Save the augmented dataframe (with SDI columns) as 
  `results/data/sdi_data.csv` for downstream layers to consume.

## Common Pitfalls

- **Computing SDI on labels instead of scores**: SDI is on continuous scores. 
  Cohen's kappa is on labels. Don't mix them.
- **Forgetting that LLM score and FinBERT score have different distributions**: 
  LLM scores tend to be more polarized (closer to ±1). This is fine — SDI 
  captures the disagreement regardless. But noting this in the paper as a 
  Limitation is worth doing.
- **Overcounting in 'ambiguous' quadrant**: If er > hi, classify as ambiguous 
  regardless of le. Don't double-count.
