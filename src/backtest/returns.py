"""Forward price returns from SQLite prices table.

ret_k[t] = log(close[t+k] / close[t])  — cumulative k-day forward return.
daily_ret[t] = log(close[t] / close[t-1]) — used for lead-lag analysis.

All returns are expressed as log returns (continuously compounded).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.store import DB_PATH


def build_returns(
    tickers: list[str] | None = None,
    db_path: Path = DB_PATH,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Compute forward log returns for each (ticker, trading_day).

    Args:
        tickers:  Ticker symbols. None = all tickers in the prices table.
        db_path:  SQLite file path.
        horizons: Forward horizons in trading days (default: 1, 3, 5).

    Returns:
        DataFrame with columns: ticker, date, daily_ret, ret_1, ret_3, ret_5.
        NaN for rows where there are insufficient future trading days.
    """
    conn = sqlite3.connect(db_path)
    if tickers:
        ph = ",".join("?" * len(tickers))
        rows = conn.execute(
            f"SELECT ticker, date, close FROM prices WHERE ticker IN ({ph}) ORDER BY ticker, date",
            tickers,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ticker, date, close FROM prices ORDER BY ticker, date"
        ).fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])

    parts: list[pd.DataFrame] = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        closes = grp["close"].values.astype(float)
        n = len(closes)

        result: dict = {"ticker": ticker, "date": grp["date"].values, "close": closes}

        # Daily log return (on day t, uses close[t-1])
        daily_ret = np.full(n, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            daily_ret[1:] = np.log(closes[1:] / closes[:-1])
        result["daily_ret"] = daily_ret

        # Forward cumulative log returns
        for h in horizons:
            fwd = np.full(n, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                valid = (closes[:n - h] > 0) & (closes[h:] > 0)
                fwd[: n - h][valid] = np.log(closes[h:][valid] / closes[: n - h][valid])
            result[f"ret_{h}"] = fwd

        parts.append(pd.DataFrame(result))

    return pd.concat(parts, ignore_index=True)
