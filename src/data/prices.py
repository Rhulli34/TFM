"""Fetch daily OHLCV prices via yfinance."""

import yfinance as yf
import pandas as pd


def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily adjusted closing prices for *ticker*.

    Args:
        ticker: Yahoo Finance symbol (e.g. "AAPL").
        start:  Start date, inclusive, "YYYY-MM-DD".
        end:    End date, exclusive,   "YYYY-MM-DD".

    Returns:
        DataFrame with a DatetimeIndex and columns
        [Open, High, Low, Close, Volume, Dividends, Stock Splits].
        Raises ValueError if the download returns no data.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker} [{start} → {end}]")
    # Flatten multi-level columns produced by yfinance ≥ 0.2.x
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
