# TriAgent: Divergence-Aware Multi-Agent Committee for Financial Sentiment

## What We're Building

A research codebase for an IJCAI 2026 workshop paper (FinLLM@IJCAI 2026).

**Core thesis** (updated 2026-04-30 — see `docs/10_AGENTIC_PIVOT.md`):
We ask **how small the agents in an agentic financial system can get**.
A three-tier committee (VADER → FinBERT → local LLM, swept across
Qwen2.5-{0.5B, 1.5B, 3B, 7B, 14B-4bit}) is orchestrated by (a) a Semantic
Divergence Index that routes by query difficulty, and (b) disagreement-
triggered **interaction protocols** (vote / critic / debate). The headline
question: **can N small models interacting outperform a single large model
at the same total compute?**

Original sub-thesis (still holds): production agentic financial systems
default to the largest model for every query, wasting GPU-hours on trivially
classifiable inputs. SDI-routed committees achieve 95% of the largest
model's accuracy at ~15% of the cost.

## Target Venue

- FinLLM@IJCAI 2026 (International Symposium on LLMs for Financial Services),
  Bremen, Germany, August 15-17, 2026
- Workshop site: https://finllm.github.io/workshop/#/
- EasyChair: https://easychair.org/conferences/?conf=finllmijcai2026
- **Submission deadline: 2026-05-15 23:59:59 AoE** (extended from May 1)
- Notification: 2026-06-01
- 7 pages + 2 refs, single-blind, IJCAI 2026 LaTeX template, EasyChair submission
- ≥1 author must travel to Bremen — **Isabel covers in-person attendance**
- **Industry-led workshop** (E Fund Management is host;
  contact yanjiangpeng@efunds.com.cn) — favor concrete deployment scenarios
  over abstract theoretical novelty
- CFP explicitly calls out: cloud+edge collaboration, token economics,
  multi-agent interaction scaling, bias diversity, and the OpenClaw / OPC
  paradigm. Cite OpenClaw and OPC in the Introduction.

## Authors

- **Jiacheng Ding** — University of Memphis (lead)
- **Isabel** — Overlake High School (co-author, in-person presenter)

## Project Structure

```
triagent/
├── BRIEF.md                     # This file (master overview)
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── docs/                        # All planning docs (ALL .md files live here)
│   ├── 01_REPO_STRUCTURE.md
│   ├── 02_DATASETS_AND_BASELINES.md
│   ├── 03_EXPERIMENT_PLAN.md
│   ├── 04_LAYER_L1_SPEC.md
│   ├── 05_LAYER_L2_SPEC.md
│   ├── 06_LAYER_L3_SPEC.md
│   ├── 07_LAYER_L4_SPEC.md
│   ├── 08_LAYER_L5_SPEC.md
│   └── 09_PAPER_OUTLINE.md
│
├── data/
│   ├── raw/                     # FPB dataset, market data
│   ├── processed/               # Pipeline outputs
│   └── lexicons/                # Mined trigger n-grams
│
├── src/
│   ├── config.py
│   ├── agents/
│   │   ├── base.py              # SentimentAgent ABC + AgentOutput dataclass
│   │   ├── vader_agent.py       # L1
│   │   ├── finbert_agent.py     # L2 (batched GPU inference)
│   │   └── llm_agent.py         # L3 (full token tracking)
│   ├── metrics/
│   │   ├── sdi.py               # Three-way SDI
│   │   ├── bias_diversity.py    # Cohen's kappa + entropy
│   │   └── cost.py              # Token cost utilities
│   ├── routing/
│   │   ├── strategies.py        # 7 routing strategies (S0-S6)
│   │   └── pareto.py            # Pareto frontier
│   ├── mining/
│   │   ├── log_odds.py          # ngram_range=(1,3) version
│   │   └── ngram_features.py    # Trigger ngram extraction
│   ├── predictor/
│   │   ├── features.py          # Edge-side feature extraction
│   │   ├── train.py             # LR + XGBoost
│   │   └── evaluate.py          # AUC / P@R metrics
│   ├── backtest/
│   │   ├── engine.py            # NO synthetic signals — real model preds only
│   │   ├── strategies.py        # Trading strategies
│   │   └── metrics.py           # Sharpe, MaxDD, Excess Return per $
│   ├── viz/
│   │   ├── style.py             # Unified plot style
│   │   └── figures.py           # Paper figures
│   └── utils/
│       └── data_loader.py       # FPB loading with HF + Kaggle fallbacks
│
├── experiments/
│   ├── L1_data_collection.py    # Foundation: produces committee_data.csv
│   ├── L2_sdi_analysis.py       # Three-way SDI + bias diversity
│   ├── L3_routing_pareto.py     # ⭐ MAIN EXPERIMENT
│   ├── L4_predictor.py          # Edge predictor (with bigram features)
│   ├── L5_backtest.py           # Real backtest (20 stocks, 2023-2024)
│   ├── L6_theory.ipynb          # Light info-theoretic framing
│   └── L7_chinese_pilot.py      # Optional cross-lingual mini-experiment
│
├── results/
│   ├── data/                    # Intermediate CSVs
│   ├── figures/                 # 300dpi PNGs
│   └── tables/                  # LaTeX tables
│
└── paper/                       # ← Paper lives INSIDE the project
    ├── main.tex
    ├── references.bib
    ├── sections/
    │   ├── 01_introduction.tex
    │   ├── 02_related_work.tex
    │   ├── 03_framework.tex
    │   ├── 04_mining.tex
    │   ├── 05_experiments.tex
    │   ├── 06_discussion.tex
    │   └── 07_conclusion.tex
    └── figures/                 # Symlink to ../results/figures
```

## 8-Layer Experiment Plan

| Layer | What | Output | Status |
|-------|------|--------|--------|
| L1   | Three-agent data collection (Qwen-7B as L3) | `committee_data.csv` | running 4/30 |
| **L1.5** | **Multi-size LLM sweep (Qwen 0.5B/1.5B/3B/7B/14B-4bit)** | size-extended data | NEW |
| L2   | Three-way SDI + Bias Diversity | Figures + quadrant table | TODO |
| **L2.5** | **Interaction protocols: vote / critic / debate** | interaction_results.csv | NEW |
| L3   | Cost-Pareto with size axis | ⭐ Main figure | TODO |
| **L3.5** | **Scaling-inflection + interaction-vs-size figures** | ⭐ Second figure | NEW |
| L4   | Edge-side predictor (uni+bigram) | Deployable router | TODO |
| L5   | End-to-end backtest (no synthetic signals!) | Equity curves | TODO |
| L6   | Light theoretical framing | Math sections | TODO |
| L7   | Optional cross-lingual pilot | Bonus material | TODO |
| L8   | Paper writing + figures | PDF submission | TODO |

**Rule**: Do not start layer N+1 until layer N's checkpoint passes.

## Three Agents

| Agent | Role | Reference Acc/F1 | Latency | Cost |
|-------|------|------------------|---------|------|
| VADER | L1 (cheap edge) | 54.3% / 48.9% | 0.1ms | $0 |
| FinBERT (ProsusAI/finbert) | L2 (specialist) | 88.9% / 88.2% | 3.6ms | ~$0 (GPU) |
| GPT-4o-mini | L3 (reasoner) | ~84% / ~84% | 1245ms | $0.0003/query |

## Routing Strategies

- S0: Always-L1, S1: Always-L2, S2: Always-L3 (degenerate baselines)
- S3: Random-x% escalation
- S4: Confidence-based (escalate VADER's low-confidence samples)
- S5: SDI-LE-x% (our basic — escalate top SDI samples)
- S6: SDI-Two-Stage (our full — L1→L2 by SDI_LE, L2→L3 by SDI_ER) ⭐⭐

## Key Innovation: Bigram Trigger Features

Single words like "loss" are ambiguous. Bigrams disambiguate:
- "loss_narrowed" → positive
- "loss_widened" → negative
- "profit_declined" → negative
- "profit_rose" → positive

L4's edge predictor must use ngram_range=(1, 3) features.

## Reproducibility Checkpoints

After L1, baseline numbers must reproduce within 1pp:
```
VADER:   Acc≈0.5433  F1≈0.4889  Latency≈0.1ms
FinBERT: Acc≈0.8894  F1≈0.8822  Latency≈3.6ms (batched)
LLM:     Acc≈0.84    F1≈0.84    Latency≈1245ms
```

If numbers deviate, STOP and debug before proceeding to L2.

## What NOT to Do (Fixing Original Code's Issues)

1. **No synthetic backtest signals.** L5 must use real model predictions.
2. **No "Type A/B/C" error taxonomy.** Drop it.
3. **No standalone "semantic shift dictionary" artifact.** Mined trigger words
   should be features for L4's predictor.
4. **No speculation about real-time volatility prediction.** Future work only.

## How to Use These Docs (When Vibe Coding)

1. Always read `BRIEF.md` (this file) first for the high-level picture.
2. Before implementing a layer, read its dedicated spec:
   - L1 → `docs/04_LAYER_L1_SPEC.md`
   - L2 → `docs/05_LAYER_L2_SPEC.md`
   - L3 → `docs/06_LAYER_L3_SPEC.md`
   - L4 → `docs/07_LAYER_L4_SPEC.md`
   - L5 → `docs/08_LAYER_L5_SPEC.md`
3. After completing a layer, verify checkpoints in the spec before moving on.
4. Paper writing references `docs/09_PAPER_OUTLINE.md`.

## Current Status

- [x] Planning docs written (this folder)
- [ ] L1: Three-agent data collection
- [ ] L2: Three-way SDI + Bias Diversity
- [ ] L3: Token-Economic Routing Pareto
- [ ] L4: Edge-side predictor
- [ ] L5: End-to-end backtest
- [ ] L6: Theoretical framing
- [ ] L7: Cross-lingual pilot (optional)
- [ ] L8: Paper writing
