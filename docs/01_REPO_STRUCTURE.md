# Repository Structure (Detailed)

## Naming Conventions

- All Python files snake_case
- All experiment scripts prefixed `L<n>_` for ordering
- All output CSVs in `results/data/`, all figures in `results/figures/`
- Figure naming: `fig_<purpose>_<descriptor>.png` (e.g. `fig_main_pareto.png`)

## Files NOT to Commit

- `.env` (real API keys)
- `data/raw/market_data/` (yfinance cache, regenerable)
- `results/` (regenerable from experiments)
- `__pycache__/`, `*.pyc`
- `paper/main.pdf` (generated)

## Module Responsibilities

### `src/config.py`

- Path constants (DATA_DIR, RESULTS_DIR, etc.)
- API keys via `os.getenv` (loaded from `.env`)
- Model name constants (LLM_MODEL = "gpt-4o-mini")
- LLM_PRICING dict (per 1M tokens)
- SDI thresholds (SDI_HIGH=0.7, SDI_LOW=0.3)
- Label maps (LABEL_MAP, LABELS_ORDERED)
- DEVICE detection (cuda or cpu)

### `src/agents/`

Each agent is a class implementing `predict(text)` and `predict_batch(texts)`,
returning standardized `AgentOutput` dataclass with: `label`, `score`, 
`confidence`, `latency_ms`, `cost_usd`, `extra`.

### `src/metrics/`

Pure functions over dataframes. No side effects. Always operate on the 
canonical `committee_data.csv` schema.

### `src/routing/`

`strategies.py` contains 7 routing functions, each takes a dataframe and 
config (escalation_pct, thresholds), returns `(predictions, total_cost, 
total_latency, coverage_breakdown)`.

`pareto.py` sweeps strategies across escalation percentages and computes 
the Pareto frontier.

### `src/predictor/`

`features.py`: extract ~25 edge-side features per sentence (VADER outputs, 
sentence stats, unigram/bigram/trigram trigger flags).

`train.py`: trains LR and XGBoost models with stratified 80/20 split.

`evaluate.py`: AUC-ROC, precision-recall curves, feature importance.

### `src/backtest/`

`engine.py`: takes a sequence of (date, signal) pairs and a price dataframe,
simulates trades with T+1 execution, slippage, holding period.

`strategies.py`: 7 trading strategies (Buy-Hold, Oracle, Always-L1/L2/L3, 
SDI-Single, SDI-Two-Stage).

`metrics.py`: Sharpe, MaxDD, Win Rate, Excess Return per Inference Dollar.

### `src/viz/`

`style.py`: matplotlib rcParams for IJCAI-friendly figures (300dpi, sans-serif,
no over-styling).

`figures.py`: factory functions for each named figure in the paper.

## Workflow

```
1. Read BRIEF.md and the relevant L<n> spec
2. Implement src/ modules referenced by the spec
3. Implement experiments/L<n>_<name>.py
4. Run with --sample 100 first to validate
5. Run full
6. Verify checkpoint criteria from the spec
7. Move to next layer
```

## Where the Paper Lives

`paper/` is **inside** the project, version-controlled together with code.
- `main.tex` is the IJCAI 2026 entry point
- Section sources in `paper/sections/`
- Figures pulled in from `paper/figures/` (symlink to `../results/figures/`)

This keeps everything in one git repo. When submitting, build `main.pdf` and
upload to EasyChair.

## Tooling

- Python 3.10+
- Use `dataclasses` for structured outputs (no dict-typing)
- Use `tqdm` for progress bars
- Use `argparse` for CLI args (always support `--sample N` for quick testing)
- Use `pathlib.Path` for paths (not raw strings)
- Use `python-dotenv` for env vars
