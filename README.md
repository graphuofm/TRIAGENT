# TriAgent

Divergence-Aware Multi-Agent Committee for Financial Sentiment Analysis.

Companion code for the FinLLM @ IJCAI 2026 workshop submission.

## TL;DR

A three-tier sentiment committee — **VADER** (lexicon, edge) →
**FinBERT** (domain transformer) → **GPT-4o-mini** (reasoning LLM) — routed
by a **Semantic Divergence Index (SDI)**. Goal: 95% of GPT-4o-mini's accuracy
at ~15% of the cost.

See [`BRIEF.md`](BRIEF.md) for the full project overview and
[`docs/`](docs/) for layer-by-layer specs.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and put in your OPENAI_API_KEY
```

## Running L1 (Foundation Data Collection)

```bash
# Sample test, no API cost
python experiments/L1_data_collection.py --sample 100 --skip-llm

# Sample test with LLM (~$0.05)
python experiments/L1_data_collection.py --sample 100

# Full run on all 4,846 FPB sentences (~$2-3 LLM cost)
python experiments/L1_data_collection.py --yes
```

Output → `results/data/committee_data.csv`.

## Layered Plan

| Layer | What | Run | Output |
|-------|------|-----|--------|
| L1 | Three-agent data collection | `experiments/L1_data_collection.py` | `committee_data.csv` |
| L2 | Three-way SDI + bias diversity | `experiments/L2_sdi_analysis.py` | `sdi_data.csv`, figures |
| L3 | Token-economic Pareto routing ⭐ | `experiments/L3_routing_pareto.py` | main figure |
| L4 | Edge-side predictor (uni+bigram) | `experiments/L4_predictor.py` | deployable router |
| L5 | End-to-end backtest (real signals) | `experiments/L5_backtest.py` | equity curves |
| L6 | Light theoretical framing | `experiments/L6_theory.ipynb` | math sections |
| L7 | Cross-lingual pilot (optional) | `experiments/L7_chinese_pilot.py` | bonus material |
| L8 | Paper writing | `paper/main.tex` | PDF |

**Rule**: do not start layer N+1 until layer N's checkpoint passes (see
each layer's spec in `docs/`).

## Repo Layout

See [`BRIEF.md`](BRIEF.md). Code in `src/`, runnable experiments in
`experiments/`, planning docs in `docs/`, figures and CSVs land under
`results/`.
