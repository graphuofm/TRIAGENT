"""L5: End-to-end backtest with REAL model predictions (no synthetic signals).

Each strategy gets a different `label_col` from sdi_data.csv:
    Buy-and-Hold   -> benchmark, no signal
    Oracle         -> 'label_text' (gold)
    Always-L1      -> 'vader_label'
    Always-L2      -> 'finbert_label'
    Always-L3      -> 'llm_label'
    SDI single     -> use S5 routing (already in pareto_points.csv via routing module)
    SDI two-stage  -> use S6 routing (Balanced operating point)
    Critic         -> 'critic_label' from interaction_results

Each strategy is evaluated on 20 tickers × 2-year period.
Headline metric: Excess Return per Inference Dollar.

Usage:
    python experiments/L5_backtest.py
    python experiments/L5_backtest.py --tickers AAPL,TSLA  # quick subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DATA_DIR, FIGURES_DIR, TABLES_DIR
from src.backtest.data import (
    fetch_all, ALL_TICKERS, SECTOR_OF, DEFAULT_START, DEFAULT_END,
)
from src.backtest.engine import run_backtest, buy_and_hold, BacktestConfig
from src.backtest.strategies import build_weekly_signals
from src.backtest.metrics import summarise
from src.routing.strategies import (
    route_always_l1, route_always_l2, route_always_l3,
    route_sdi_pct, route_two_stage,
)
from src.viz.style import apply_style, COLORS


def load_strategy_predictions(sdi: pd.DataFrame) -> dict[str, pd.Series]:
    """Materialise per-strategy label vectors over the sentence corpus.

    Routing strategies (S5, S6) are computed on the fly from sdi_data.csv;
    interaction strategies (critic, debate) are pulled from the
    `interaction_results_*` CSVs if present.
    """
    out: dict[str, pd.Series] = {
        'Oracle':            sdi['label_text'],
        'Always-L1':         sdi['vader_label'],
        'Always-L2':         sdi['finbert_label'],
        'Always-L3':         sdi['llm_label'],
    }
    # SDI single-stage (S5 with 30% L1->L2 escalation, balanced op-point)
    s5 = route_sdi_pct(sdi, escalation_pct=0.30, sdi_col='sdi_le')
    out['SDI-Single (S5)'] = pd.Series(s5.predictions, index=sdi.index)

    # SDI two-stage (S6 Balanced: 30% L2, 5% L3)
    s6 = route_two_stage(sdi, pct_l1_to_l2=0.30, pct_l2_to_l3=0.05)
    out['SDI-Two-Stage (S6)'] = pd.Series(s6.predictions, index=sdi.index)

    # Interaction protocols — pull whatever's been run
    for path in sorted(RESULTS_DATA_DIR.glob('interaction_results_*.csv')):
        stem = path.stem.replace('interaction_results_', '')
        try:
            inter = pd.read_csv(path)
        except Exception:
            continue
        for col in ['vote_label', 'critic_label', 'debate_label']:
            if col not in inter.columns:
                continue
            tag = col.replace('_label', '')
            # Align by sentence_id where possible
            if 'sentence_id' in inter.columns and 'sentence_id' in sdi.columns:
                m = sdi[['sentence_id']].merge(
                    inter[['sentence_id', col]], on='sentence_id', how='left')
                out[f'{tag}@{stem}'] = m[col]
    return out


def per_strategy_inference_cost(sdi: pd.DataFrame, strategies: list[str],
                                 n_signals_per_week: int,
                                 n_weeks: int, n_tickers: int) -> dict[str, float]:
    """Approximate the total $-cost of each strategy's signal generation
    across the backtest window. Uses per-sample cost columns in sdi_data.csv
    plus the per-protocol summary CSVs for interaction protocols.
    """
    n_signals = n_signals_per_week * n_weeks * n_tickers
    vader_pcost   = sdi['vader_cost_usd'].mean()
    finbert_pcost = sdi['finbert_cost_usd'].mean()
    llm_pcost     = sdi['llm_cost_usd'].mean() if 'llm_cost_usd' in sdi.columns else 0.0

    # Approximate critic@<size> by reading the per-protocol summary
    def _critic_extra_pcost(suffix: str) -> float:
        """Avg extra cost per sentence for the critic call at this size."""
        path = RESULTS_DATA_DIR / f'interaction_summary_critic_{suffix}.csv'
        if not path.exists():
            return 0.0
        df_sum = pd.read_csv(path)
        row = df_sum[df_sum['protocol'] == 'critic']
        if len(row) == 0:
            return 0.0
        # Total extra cost across the full sentence corpus / corpus size
        n_sentences = len(sdi)
        return float(row['extra_cost_usd'].iloc[0]) / max(n_sentences, 1)

    base = {
        'Oracle':              0.0,
        'Always-L1':           vader_pcost   * n_signals,
        'Always-L2':           finbert_pcost * n_signals,
        'Always-L3':           llm_pcost     * n_signals,
        'SDI-Single (S5)':     (vader_pcost + finbert_pcost * 0.30) * n_signals,
        'SDI-Two-Stage (S6)':  (vader_pcost + finbert_pcost * 0.30
                                + llm_pcost * 0.30 * 0.05) * n_signals,
    }

    # Add costs for any interaction strategies present
    for s in strategies:
        if s in base:
            continue
        # Names look like 'critic@1p5b', 'critic@critic_1p5b_tfns', 'debate@7b' etc.
        if '@' in s:
            tag, suffix = s.split('@', 1)
            # Strip new-style 'critic_' or 'debate_' prefix on suffix
            suffix = suffix.replace('critic_', '').replace('debate_', '')
            if tag in ('critic', 'debate'):
                # V + F always run + extra LLM call on triggered samples
                extra_per_sample = _critic_extra_pcost(suffix)
                base[s] = (vader_pcost + finbert_pcost + extra_per_sample) * n_signals
            else:
                base[s] = 0.0
        else:
            base[s] = 0.0
    return base


def main(args):
    apply_style()

    sdi_path = Path(args.input) if args.input else RESULTS_DATA_DIR / 'sdi_data.csv'
    if not sdi_path.exists():
        raise FileNotFoundError(f"sdi_data file not found: {sdi_path}")
    sdi = pd.read_csv(sdi_path)
    print(f"Loaded {len(sdi)} sentences from {sdi_path}")

    tickers = (args.tickers.split(',') if args.tickers else ALL_TICKERS)
    print(f"\nFetching market data for {len(tickers)} tickers ({args.start} to {args.end})...")
    prices_by_ticker = fetch_all(tickers, start=args.start, end=args.end)
    print(f"  got {len(prices_by_ticker)} usable tickers")

    # Friday end-of-week dates over the backtest window
    weekly_dates = pd.date_range(start=args.start, end=args.end, freq='W-FRI')
    n_weeks = len(weekly_dates)
    print(f"  {n_weeks} trading weeks")

    # Build per-strategy signals once (reused across tickers)
    strategy_preds = load_strategy_predictions(sdi)
    print(f"\nStrategies to evaluate: {list(strategy_preds.keys())}")

    cfg = BacktestConfig()
    all_summaries = []
    all_equity_curves: dict[str, list[pd.Series]] = {s: [] for s in strategy_preds}
    all_equity_curves['Buy-and-Hold'] = []

    inference_cost_per_strategy = per_strategy_inference_cost(
        sdi, strategies=list(strategy_preds.keys()),
        n_signals_per_week=args.sentences_per_week,
        n_weeks=n_weeks, n_tickers=len(prices_by_ticker))

    for ticker in prices_by_ticker:
        prices = prices_by_ticker[ticker]
        bh = buy_and_hold(prices, ticker=ticker, cfg=cfg)
        bh_ret = bh['summary']['total_return_pct']
        all_equity_curves['Buy-and-Hold'].append(bh['equity'])

        for strat_name, label_series in strategy_preds.items():
            if label_series is None or label_series.isna().all():
                continue
            preds_df = sdi[['sentence']].copy()
            preds_df['__label__'] = label_series.values
            signals = build_weekly_signals(
                preds_df, label_col='__label__',
                weekly_dates=weekly_dates,
                n_per_week=args.sentences_per_week,
                seed=42)
            res = run_backtest(prices, signals, ticker=ticker, cfg=cfg)
            all_equity_curves[strat_name].append(res['equity'])
            inf_cost = inference_cost_per_strategy.get(strat_name, 0.0)
            all_summaries.append({
                'ticker': ticker, 'sector': SECTOR_OF.get(ticker, 'unknown'),
                **summarise(strat_name, res['equity'], res['trades'],
                            inf_cost / max(len(prices_by_ticker), 1), bh_ret,
                            cfg.initial_capital),
            })

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(RESULTS_DATA_DIR / 'backtest_results.csv', index=False)

    # Aggregate across tickers per strategy
    agg = (summary_df
           .groupby('strategy')
           .agg({'total_return_pct': ['mean', 'std'],
                 'sharpe':           ['mean'],
                 'max_drawdown_pct': ['mean'],
                 'win_rate':         ['mean'],
                 'n_trades':         ['mean'],
                 'total_inference_$':['mean'],
                 'excess_per_$':     ['mean']})
           .round(4))
    agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
    agg = agg.reset_index()
    print("\n=== Aggregate (mean across tickers) ===")
    print(agg.to_string(index=False))
    agg.to_csv(RESULTS_DATA_DIR / 'backtest_summary_aggregate.csv', index=False)

    # Equity curves figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    color_map = {
        'Buy-and-Hold':      ('#000000', ':',  1.5),
        'Oracle':            ('#2ca02c', '--', 1.5),
        'Always-L1':         ('#bbbbbb', '-',  1.2),
        'Always-L2':         ('#1f77b4', '-',  1.4),
        'Always-L3':         ('#0a3a73', '-',  1.4),
        'SDI-Single (S5)':   ('#ff7f0e', '-',  1.6),
        'SDI-Two-Stage (S6)': ('#d62728', '-',  2.0),
    }
    for strat_name, curves in all_equity_curves.items():
        if not curves:
            continue
        # Align all curves on a common index
        merged = pd.concat([c.rename(i) for i, c in enumerate(curves)],
                           axis=1).sort_index().ffill()
        mean = merged.mean(axis=1)
        c, ls, lw = color_map.get(strat_name, ('#7f7f7f', '-', 1.0))
        ax.plot(mean.index, mean.values, color=c, ls=ls, lw=lw, label=strat_name, alpha=0.9)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean equity ($)')
    ax.set_title('Backtest equity curves (mean across tickers)')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    out_fig = FIGURES_DIR / 'fig_equity_curves.png'
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"  ✓ saved {out_fig}")

    # LaTeX table
    with open(TABLES_DIR / 'backtest_summary.tex', 'w') as f:
        f.write("% Auto-generated by experiments/L5_backtest.py\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Strategy & Return\\% & Sharpe & MaxDD\\% & Win\\% & \\$ Cost & Excess R/\\$ \\\\\n\\midrule\n")
        for _, r in agg.iterrows():
            strat = r['strategy'].replace('_', r'\_').replace('&', r'\&')
            f.write(
                f"{strat} & {r['total_return_pct_mean']*100:.1f} & "
                f"{r['sharpe_mean']:.2f} & "
                f"{r['max_drawdown_pct_mean']*100:.1f} & "
                f"{r['win_rate_mean']*100:.1f} & "
                f"{r['total_inference_$_mean']:.2f} & "
                f"{r['excess_per_$_mean']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  ✓ saved {TABLES_DIR / 'backtest_summary.tex'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=str, default=None,
                        help='Comma-separated ticker subset (default: all 20)')
    parser.add_argument('--start', type=str, default=DEFAULT_START)
    parser.add_argument('--end',   type=str, default=DEFAULT_END)
    parser.add_argument('--sentences-per-week', type=int, default=20)
    parser.add_argument('--input', type=str, default=None,
                        help='Path to sdi_data.csv (default: results/data/sdi_data.csv). '
                             'Use sdi_data_fpb_backup.csv for FPB-only paper-quality runs.')
    args = parser.parse_args()
    main(args)
