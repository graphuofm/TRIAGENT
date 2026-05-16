"""Single-ticker backtest engine for the L5 experiment.

Trade rules (per docs/08 spec):
  - Signal date = end-of-week (Friday close)
  - Execution    = next Monday open (T+1)
  - Hold         = 5 trading days (close-to-close)
  - Position     = 10% of capital per trade
  - Slippage     = 10 bps each side (fill = open*(1+slip), exit = close*(1-slip))
  - One position per ticker at a time; if a new signal arrives mid-hold, ignore
    (no overlapping positions, conservative).

Signals: BUY (long), SHORT (short), HOLD (no trade).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


Signal = Literal['BUY', 'SHORT', 'HOLD']


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    position_frac:   float = 0.10
    hold_days:       int   = 5
    slippage_bps:    float = 10.0    # round-trip slippage (10 bps each side)


@dataclass
class Trade:
    ticker: str
    signal_date: pd.Timestamp
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    direction:   Signal
    entry_price: float
    exit_price:  float
    pnl_usd:     float
    pnl_pct:     float


def run_backtest(
    prices: pd.DataFrame,
    signals: list[tuple[pd.Timestamp, Signal]],
    *,
    ticker: str,
    cfg: BacktestConfig | None = None,
) -> dict:
    """Run a single-ticker backtest.

    Args:
        prices: DataFrame with ['Open','High','Low','Close','Volume'], date index.
        signals: list of (signal_date, signal). signal_date should be a Friday close.
        ticker:  ticker symbol (used in trade records only).
        cfg:     BacktestConfig or None for defaults.

    Returns:
        dict with keys 'trades' (DataFrame), 'equity' (Series), 'summary' (dict).
    """
    cfg = cfg or BacktestConfig()
    prices = prices.sort_index()
    capital = cfg.initial_capital
    equity_dates: list[pd.Timestamp] = [prices.index.min()]
    equity_vals:  list[float] = [capital]
    trades: list[Trade] = []

    in_position_until: pd.Timestamp | None = None

    slip = cfg.slippage_bps / 10_000.0

    # Pre-index price dates for fast next-trading-day lookup
    pdates = prices.index

    for sdate, sig in signals:
        if sig == 'HOLD':
            continue
        # Skip if still holding from a previous trade
        if in_position_until is not None and sdate <= in_position_until:
            continue

        # Find next trading day on or after sdate+1
        idx = pdates.searchsorted(sdate + pd.Timedelta(days=1))
        if idx >= len(pdates) - cfg.hold_days:
            break  # not enough days to complete a hold
        entry_date = pdates[idx]
        exit_date  = pdates[idx + cfg.hold_days - 1]

        entry_price_raw = float(prices.loc[entry_date, 'Open'])
        exit_price_raw  = float(prices.loc[exit_date,  'Close'])

        if sig == 'BUY':
            entry_price = entry_price_raw * (1 + slip)
            exit_price  = exit_price_raw  * (1 - slip)
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            entry_price = entry_price_raw * (1 - slip)
            exit_price  = exit_price_raw  * (1 + slip)
            pnl_pct = (entry_price - exit_price) / entry_price

        pos_size = capital * cfg.position_frac
        pnl_usd  = pos_size * pnl_pct
        capital += pnl_usd

        trades.append(Trade(ticker=ticker, signal_date=sdate, entry_date=entry_date,
                            exit_date=exit_date, direction=sig,
                            entry_price=entry_price, exit_price=exit_price,
                            pnl_usd=pnl_usd, pnl_pct=pnl_pct))
        equity_dates.append(exit_date)
        equity_vals.append(capital)
        in_position_until = exit_date

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates),
                       name='equity').sort_index()

    return {
        'trades':  trades_df,
        'equity':  equity,
        'summary': {
            'ticker':           ticker,
            'final_capital':    float(capital),
            'total_return_pct': float((capital - cfg.initial_capital) / cfg.initial_capital),
            'n_trades':         int(len(trades)),
            'n_winning':        int((trades_df['pnl_usd'] > 0).sum()) if len(trades_df) else 0,
        },
    }


def buy_and_hold(prices: pd.DataFrame, *, ticker: str,
                  cfg: BacktestConfig | None = None) -> dict:
    """Buy at the first close, hold to the last close. Used as the benchmark."""
    cfg = cfg or BacktestConfig()
    prices = prices.sort_index()
    if len(prices) < 2:
        return {'trades': pd.DataFrame(), 'equity': pd.Series(dtype=float),
                'summary': {'ticker': ticker, 'total_return_pct': 0.0,
                            'final_capital': cfg.initial_capital, 'n_trades': 0}}
    entry = float(prices['Close'].iloc[0])
    exit_ = float(prices['Close'].iloc[-1])
    pnl_pct = (exit_ - entry) / entry
    final = cfg.initial_capital * (1 + pnl_pct)
    equity = pd.Series([cfg.initial_capital, final],
                       index=[prices.index[0], prices.index[-1]], name='equity')
    return {
        'trades': pd.DataFrame(),
        'equity': equity,
        'summary': {
            'ticker': ticker, 'final_capital': float(final),
            'total_return_pct': float(pnl_pct), 'n_trades': 0,
        },
    }
