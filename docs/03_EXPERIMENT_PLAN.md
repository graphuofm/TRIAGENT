# Experiment Plan (8-Layer Architecture)

## Layer Dependencies

```
L8: Paper Writing
    ↑
L7: Cross-lingual (optional, parallel to L5)
    ↑
L6: Theoretical Framing (lightweight, ~0.3 page)
    ↑
L5: End-to-End Backtest (real signals, 20 stocks)
    ↑
L4: Edge-Side Predictor (unigram + BIGRAM + trigram features)
    ↑
L3: Token-Economic Routing (← MAIN: Pareto frontier)
    ↑
L2: Three-way SDI + Bias Diversity
    ↑
L1: Committee Data Collection (FOUNDATION)
```

**Rule**: Do not start layer N+1 until layer N's checkpoint passes.

## Time Budget

Assumes May 1 deadline; pad ~2 days for buffer. If deadline is May 15, 
you have a much more comfortable timeline.

| Day | Focus | Checkpoint |
|-----|-------|------------|
| Day 1 | Repo scaffold, L1 sample test | VADER+FinBERT reproduce |
| Day 2 | L1 full run with LLM | All 3 baselines reproduce |
| Day 3 | L2 three-way SDI | Four-quadrant table sensible |
| Day 4-5 | L3 routing Pareto | **Pareto figure looks clean** |
| Day 6 | L4 edge predictor | AUC ≥ 0.80 with bigram features |
| Day 7-8 | L5 backtest | Committee strategy beats Always-L3 on $-adjusted return |
| Day 9 | L6 theory + L7 Chinese (parallel) | Math + bonus done |
| Day 10-11 | Paper draft | Full draft circulated |
| Day 12-13 | Polish, figures, format | Ready to submit |
| Day 14 | Submit | EasyChair upload |

## Critical Path Risks

1. **L3 Pareto figure unimpressive**: If SDI-routing doesn't visibly 
   dominate confidence/random, the paper has no story.
   - Mitigation: try multiple SDI variants (sdi_le, sdi_max)
   - Tune escalation thresholds carefully
   - Try log-scale x-axis if linear doesn't show separation

2. **L4 predictor AUC < 0.78**: Without a deployable predictor, the 
   framework reduces to "post-hoc analysis" only.
   - Mitigation: add bigram + trigram features
   - Add sentence-BERT embedding of LLM reasoning as ablation

3. **L5 backtest noisy**: Single-stock or single-period results may not 
   show committee advantage clearly.
   - Mitigation: aggregate across 20 stocks, report mean ± std
   - Report Excess Return per Inference Dollar (cost-adjusted)

## Per-Layer Output Manifest

| Layer | Code Output | Data Output | Figure Output |
|-------|-------------|-------------|---------------|
| L1 | `experiments/L1_data_collection.py` | `committee_data.csv` | — |
| L2 | `experiments/L2_sdi_analysis.py` | `sdi_quadrants.csv` | `fig_sdi_three_way.png`, `fig_bias_diversity.png` |
| L3 | `experiments/L3_routing_pareto.py` | `pareto_points.csv`, `operating_points.tex` | `fig_main_pareto.png` ⭐ |
| L4 | `experiments/L4_predictor.py` | `predictor_results.csv` | `fig_predictor_auc.png`, `fig_feature_importance.png` |
| L5 | `experiments/L5_backtest.py` | `backtest_results.csv`, `backtest_summary.tex` | `fig_equity_curves.png` |
| L6 | `experiments/L6_theory.ipynb` | — | `fig_kl_sdi_correspondence.png` |
| L7 | `experiments/L7_chinese_pilot.py` | `chinese_results.csv` | `fig_crosslingual.png` |

⭐ = the figure that determines whether the paper succeeds

## Definition of Done — Per Layer

Each L<n> spec defines its own checkpoint criteria. **Do not skip these**.
The whole point of the layered approach is to catch bugs early. If L1 has 
a bug, every downstream layer is poisoned.
