"""
data.py - Download, cache, and preprocess financial time series from yfinance.
"""

import os
import re
import numpy as np
import pandas as pd
import yfinance as yf


CACHE_DIR = "cache"
DEFAULT_START = "1990-01-01"


def load_data(
    ticker: str = "^MXX",
    start_date: str = DEFAULT_START,
    cutoff_date: str = "2022-01-01",
    test_days: int = 252,
    horizon: int = 10,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """
    Returns a DataFrame with DatetimeIndex and 'Close' column covering
    [start_date, cutoff_date + test_days + horizon buffer].

    Cache strategy:
      - Cache files are named  {ticker}_{start}_{end}.parquet
      - Before downloading, scans for any existing file for this ticker that
        fully covers the requested [start, end] range; if found, loads and
        filters it instead of hitting the network.
      - Otherwise downloads from yfinance and saves a new cache file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_dt = pd.Timestamp(start_date)
    cutoff_dt = pd.Timestamp(cutoff_date)
    # Convert trading days to calendar days with buffer (≈ ×1.4 to account for weekends/holidays)
    calendar_days = int(test_days * 1.4) + horizon * 2
    end_dt = cutoff_dt + pd.DateOffset(days=calendar_days)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    ticker_slug = ticker.replace("^", "").replace("/", "-")
    exact_cache = os.path.join(cache_dir, f"{ticker_slug}_{start_str}_{end_str}.parquet")

    # 1. Exact cache hit
    if os.path.exists(exact_cache):
        print(f"Loading cached data from {exact_cache}")
        return pd.read_parquet(exact_cache)

    # 2. Look for any existing cache for this ticker that covers the range
    covering = _find_covering_cache(cache_dir, ticker_slug, start_dt, end_dt)
    if covering:
        print(f"Using existing cache {os.path.basename(covering)} (covers requested range)")
        df = pd.read_parquet(covering)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()
        # Save as the exact-match file so future runs are faster
        df.to_parquet(exact_cache)
        return df

    # 3. Download from yfinance
    print(f"Downloading {ticker} from {start_str} to {end_str}...")
    df = _download_yfinance(ticker, start_str, end_str)
    df.to_parquet(exact_cache)
    print(f"Cached to {exact_cache}")
    return df


def _find_covering_cache(
    cache_dir: str,
    ticker_slug: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> str | None:
    """
    Scans cache_dir for parquet files named {ticker_slug}_{start}_{end}.parquet
    (new format) or the legacy {ticker_slug}_{end}.parquet format, and returns
    the path of the first file whose stored date range fully covers [start_dt, end_dt].
    Returns None if no suitable file is found.
    """
    pattern_new = re.compile(
        rf"^{re.escape(ticker_slug)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})\.parquet$"
    )
    pattern_legacy = re.compile(
        rf"^{re.escape(ticker_slug)}_(\d{{4}}-\d{{2}}-\d{{2}})\.parquet$"
    )
    try:
        entries = os.listdir(cache_dir)
    except FileNotFoundError:
        return None

    for fname in entries:
        fpath = os.path.join(cache_dir, fname)
        m = pattern_new.match(fname)
        if m:
            file_start = pd.Timestamp(m.group(1))
            file_end = pd.Timestamp(m.group(2))
            if file_start <= start_dt and file_end >= end_dt:
                return fpath
            continue
        m = pattern_legacy.match(fname)
        if m:
            # Legacy files started from ~1990; check by loading the actual index
            try:
                df = pd.read_parquet(fpath)
                if df.index[0] <= start_dt and df.index[-1] >= end_dt:
                    return fpath
            except Exception:
                pass
    return None


def _download_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical Close prices from yfinance for [start_date, end_date).
    Uses multi_level_index=False to avoid MultiIndex columns with a single ticker.
    """
    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw[["Close"]].copy()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def split_train_test(
    df: pd.DataFrame,
    cutoff_date: str,
    test_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df into train (up to and including cutoff_date) and test (after cutoff_date).
    test_days controls how many trading days the test period covers.
    """
    cutoff_dt = pd.Timestamp(cutoff_date)
    train = df[df.index <= cutoff_dt].copy()
    test_all = df[df.index > cutoff_dt].copy()
    test = test_all.iloc[:test_days].copy()
    return train, test


def get_forecast_origins(
    test_df: pd.DataFrame,
    n_samples: int,
    horizon: int,
) -> list[pd.Timestamp]:
    """
    Returns n_samples evenly-spaced forecast origin timestamps from test_df.
    Each origin must have at least `horizon` future days remaining in test_df.
    """
    # Valid range: positions 0 to len(test_df) - horizon - 1
    max_pos = len(test_df) - horizon - 1
    if max_pos < 0:
        raise ValueError(
            f"test_df has only {len(test_df)} rows, not enough for horizon={horizon}"
        )
    n = min(n_samples, max_pos + 1)
    positions = np.linspace(0, max_pos, n, dtype=int)
    # Deduplicate while preserving order
    seen = set()
    unique_positions = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique_positions.append(p)
    return [test_df.index[p] for p in unique_positions]
