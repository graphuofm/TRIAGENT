"""Trading-performance metrics for the L5 backtest.

Standard finance metrics (Sharpe, MaxDD, WinRate) plus the headline
"Excess Return per Inference Dollar" defined in docs/08.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annualised_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe from a series of per-period returns."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    # Approximate periodicity: weekly here, but we annualise from observed step
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / std)


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown as a fraction (negative number)."""
    if len(equity) < 2:
        return 0.0
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return float(dd.min())


def win_rate(trades: pd.DataFrame) -> float:
    if len(trades) == 0:
        return 0.0
    return float((trades['pnl_usd'] > 0).mean())


def total_return(equity: pd.Series, initial_capital: float) -> float:
    if len(equity) == 0:
        return 0.0
    return float((equity.iloc[-1] - initial_capital) / initial_capital)


def excess_return_per_dollar(strategy_return: float, benchmark_return: float,
                              total_inference_cost_usd: float,
                              initial_capital: float = 100_000.0) -> float:
    """The headline metric.

    Numerator: excess $-return earned over the buy-and-hold benchmark
    Denominator: total inference dollars spent generating the signals
    """
    excess_dollars = (strategy_return - benchmark_return) * initial_capital
    if total_inference_cost_usd <= 0:
        return float('inf') if excess_dollars > 0 else float('-inf') if excess_dollars < 0 else 0.0
    return float(excess_dollars / total_inference_cost_usd)


def summarise(name: str,
              equity: pd.Series, trades: pd.DataFrame,
              total_inference_cost_usd: float,
              benchmark_return: float,
              initial_capital: float) -> dict:
    if len(equity) >= 2:
        rets = equity.pct_change().dropna()
        sharpe = annualised_sharpe(rets)
        mdd = max_drawdown(equity)
    else:
        sharpe, mdd = 0.0, 0.0
    tr = total_return(equity, initial_capital)
    return {
        'strategy':            name,
        'total_return_pct':    tr,
        'sharpe':              sharpe,
        'max_drawdown_pct':    mdd,
        'win_rate':            win_rate(trades),
        'n_trades':            int(len(trades)),
        'total_inference_$':   total_inference_cost_usd,
        'excess_per_$':        excess_return_per_dollar(
            tr, benchmark_return, total_inference_cost_usd, initial_capital),
    }
