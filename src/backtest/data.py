"""Market data loader for the L5 backtest.

Pulls daily OHLCV via yfinance for the 20 tickers in the spec,
caches each ticker as `data/raw/market_data/{ticker}.csv`.

We deliberately do NOT pin a yfinance version; if yfinance hits a
schema break in the future, the cache files protect previously-run
experiments from being invalidated.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import RAW_DIR


MARKET_CACHE = RAW_DIR / "market_data"
MARKET_CACHE.mkdir(parents=True, exist_ok=True)

# 20 tickers across 4 sectors (per docs/02 / docs/08 spec)
TICKERS_BY_SECTOR = {
    'tech':       ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META'],
    'financial':  ['JPM', 'BAC', 'GS', 'V', 'MA'],
    'consumer':   ['WMT', 'KO', 'PG', 'JNJ', 'PFE'],
    'industrial': ['XOM', 'CVX', 'BA'],
}
ALL_TICKERS = sum(TICKERS_BY_SECTOR.values(), [])
SECTOR_OF = {t: s for s, ts in TICKERS_BY_SECTOR.items() for t in ts}

DEFAULT_START = '2023-01-01'
DEFAULT_END   = '2024-12-31'


def fetch_one(ticker: str, start: str = DEFAULT_START, end: str = DEFAULT_END,
              force_refresh: bool = False) -> pd.DataFrame:
    """Fetch one ticker, cache to CSV. Returns DataFrame with date index + OHLCV."""
    cache = MARKET_CACHE / f"{ticker}.csv"
    if cache.exists() and not force_refresh:
        df = pd.read_csv(cache, parse_dates=['Date'], index_col='Date')
        return df

    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yfinance may return a MultiIndex when multiple tickers were passed; flatten if so.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    raw.index.name = 'Date'
    raw.to_csv(cache)
    return raw


def fetch_all(tickers: list[str] | None = None, **kwargs) -> dict[str, pd.DataFrame]:
    """Fetch all 20 tickers (or subset). Returns {ticker: DataFrame}."""
    tickers = tickers or ALL_TICKERS
    out = {}
    for t in tickers:
        try:
            out[t] = fetch_one(t, **kwargs)
        except Exception as e:
            print(f"  WARN: {t} failed: {e}")
    return out
