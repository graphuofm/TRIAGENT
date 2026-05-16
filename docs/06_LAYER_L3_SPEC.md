# Layer L3: Token-Economic Routing — THE CORE EXPERIMENT (思路版)

## Goal

Produce the paper's main figure: Pareto frontier of cost vs. F1-macro across
all routing strategies, demonstrating that SDI-guided two-stage routing
dominates baselines.

**This figure determines whether the paper succeeds.** Budget extra time.

## Why This Layer Matters Most

Reviewers from E Fund and other industry partners want to see one thing: 
"can I deploy this and save money?" The Pareto figure answers that 
visually in 3 seconds.

If S6 (our two-stage method) doesn't visibly dominate S3 (random) and S4 
(confidence-based), there is no headline result. Iterate until it does.

## Inputs

`results/data/sdi_data.csv` (from L2 — has all SDI columns)

## Conceptual Approach

### The Operating Points Mental Model

Don't think of routing as "one number." Think of it as a **menu of operating 
points** along a Pareto frontier:

```
       Always-L3 (right corner)  
            ●      ← expensive but accurate
                                  
        ◇ S6: Premium (90% L2, 30% L3)  ← matches Always-L3 at half cost
        ◇ S6: Balanced (30% L2, 5% L3)  ← 95% Always-L3 F1 at 15% cost  
        ◇ S6: Budget (10% L2, 1% L3)   ← 90% Always-L3 F1 at 5% cost
                                  
   ●   ● Always-L1 (left corner, cheap but bad)
```

The story: a deployment team can pick *any point on this frontier* depending 
on their budget. Our routing makes the frontier accessible.

## What to Implement

### `src/routing/strategies.py`

Seven routing strategies, each a function with signature:
```python
def route_<strategy>(df, **config) -> dict:
    """Returns {'predictions', 'total_cost', 'total_latency', 'l1_pct', 'l2_pct', 'l3_pct'}"""
```

The seven:

- **S0: Always-L1** — `df['vader_label']`
- **S1: Always-L2** — `df['finbert_label']`
- **S2: Always-L3** — `df['llm_label']`
- **S3: Random-x%** — randomly escalate `x%` from L1 to L2
- **S4: Confidence-x%** — escalate the lowest-VADER-confidence `x%` from L1 to L2
- **S5: SDI-LE-x%** — escalate the highest-SDI_LE `x%` from L1 to L2 (our basic)
- **S6: Two-Stage** — first S5 with `pct_l1_to_l2`, then escalate top SDI_ER 
  among the L2 set with `pct_l2_to_l3`

Make the strategies callable with a single config dict so the experiment script 
can sweep them uniformly.

### `src/routing/pareto.py`

For each strategy, sweep escalation percentages from 0 to 100 in 5% steps. 
For each `(strategy, pct)` point, record:
- F1-macro
- Total $ cost (USD per 1000 sentences)
- Mean latency (ms per sentence)
- Coverage breakdown (% in L1, L2, L3)

For S6 (two-stage), do a 2D sweep: outer loop over `pct_l1_to_l2 ∈ {0%, 10%, 
20%, 30%, 50%, 70%}`, inner loop over `pct_l2_to_l3 ∈ {0%, 1%, 5%, 10%, 25%}`. 
This gives ~30 points to plot.

### `experiments/L3_routing_pareto.py`

1. Load `sdi_data.csv`
2. For each strategy, run the pct sweep
3. Save all `(strategy, pct, f1, cost, latency, coverage)` tuples to 
   `results/data/pareto_points.csv`
4. Plot the Pareto figure
5. Pick three named operating points for the table

## Output Files

- `results/data/pareto_points.csv` — all points
- `results/figures/fig_main_pareto.png` — THE MAIN FIGURE ⭐
- `results/tables/operating_points.tex` — three named points table

## Pareto Figure Design Spec

Make this figure beautiful — it is the paper's first impression.

```
Y-axis: F1-Macro (linear, 0.4 to 0.95)
X-axis: $ cost per 1000 sentences (LOG SCALE — important!)

Curves:
  - Always-L1 (●, gray, single point at left)
  - Always-L2 (●, gray, single point in middle)
  - Always-L3 (●, gray, single point at right)
  - Random (—, light gray)
  - Confidence (—, medium gray)  
  - SDI-Single (-•-, blue) ⭐
  - SDI-Two-Stage (—◆—, red, thicker line) ⭐⭐

Annotations:
  - Mark the three "operating points" for SDI-Two-Stage with diamond markers
  - Add an inset zoom of the Pareto-optimal region (top-left of curve)
  
Style:
  - 300dpi
  - Sans-serif font
  - Legend in lower-right (avoid overlap with curves)
  - Grid: faint, dashed
```

The figure must communicate: **"the red curve is in the top-left corner of 
the others."**

## Operating Points Table

Pick three points along S6's curve:

| Point | Config | $/1000 | F1 | Latency | vs Always-L3 |
|-------|--------|--------|-----|---------|--------------|
| Budget | 10% L2, 1% L3 | $X | Y | Z | -3% F1, -90% cost |
| Balanced | 30% L2, 5% L3 | $X | Y | Z | -1% F1, -70% cost |
| Premium | 60% L2, 15% L3 | $X | Y | Z | match F1, -40% cost |
| Always-L3 (ref) | — | $X | Y | Z | — |

## Real-World Cost Headline (for Paper Introduction)

Compute one number that goes into the abstract/intro:

```python
queries_per_day = 100_000   # Asset manager scale
days_per_year = 252         # Trading days

l3_yearly = always_l3_cost_per_query * queries_per_day * days_per_year
balanced_yearly = balanced_cost_per_query * queries_per_day * days_per_year
savings = l3_yearly - balanced_yearly
```

Expect savings on the order of $5,000 to $40,000+/year per deployment use 
case. **Use the largest defensible number in the paper's abstract.**

## Checkpoint Criteria

Before moving to L4, verify:

1. **S6 dominates S3 (random)** at every cost level by ≥ 3 F1 points
2. **S5 outperforms S4 (confidence)** at most cost levels
3. **At "Balanced" operating point**: F1 ≥ 0.80 with cost reduction ≥ 60% 
   vs Always-L3
4. **The visual story is clear**: a non-expert looking at the figure for 5 
   seconds should see "red is best."

## Iteration Strategies (if Pareto looks unimpressive)

If S6 doesn't dominate visibly, try:

1. **Use sdi_max instead of sdi_le** for the routing signal in S6's stage 1
2. **Tune the score normalization**: if VADER and FinBERT scores have 
   wildly different scales, normalize to z-scores first
3. **Try sdi_lr for stage 1** instead of sdi_le (use the LLM as ground truth 
   for what's "hard")
4. **Use confidence as a secondary feature**: route if `sdi_le > t1 OR 
   vader_confidence < t2`
5. **Bootstrap CIs**: 1000 resamples to show the gap is statistically robust

If after all of this S6 still doesn't dominate, the story may need to shift 
to "we provide a clean Pareto frontier across multiple routing strategies, 
and SDI-based routing is competitive." That's a weaker paper but still 
publishable.

## Coding Notes

- Compute everything from `sdi_data.csv` — don't re-run agents
- The experiment is essentially "for each strategy, for each pct, simulate 
  what happens if we used those predictions" — pure dataframe operations
- Should run in < 1 minute on a laptop (no GPU/API calls)
- Save the figure with `bbox_inches='tight'` to avoid whitespace issues

## What This Layer Feeds Into

- **Paper §5.3** — the main result section
- **Paper §1** — the headline numbers in the abstract come from here
- **L4** — the predictor needs to know which sentences need escalation; uses 
  SDI thresholds defined here
