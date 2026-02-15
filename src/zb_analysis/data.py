"""Data acquisition and preprocessing helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def _to_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def _normalize_yfinance_frame(df: pd.DataFrame, local_timezone: str) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        if len(normalized.columns.levels[-1]) == 1:
            normalized.columns = normalized.columns.get_level_values(0)
        else:
            normalized.columns = normalized.columns.get_level_values(0)

    normalized = normalized.rename(columns=str.lower)
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns from Yahoo response: {missing}")

    if normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize("UTC")
    normalized.index = normalized.index.tz_convert(local_timezone).tz_localize(None)
    normalized.index.name = "date"

    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def fetch_hourly_data(
    symbol: str,
    start_date: date | datetime,
    end_date: date | datetime,
    *,
    local_timezone: str = "Asia/Dubai",
    interval: str = "60m",
    include_prepost: bool = True,
    chunk_days: int = 60,
    cache_path: Optional[Path] = None,
    fallback_csv_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly data from Yahoo Finance with chunking and optional CSV cache.

    Notes:
    - `end_date` is treated as inclusive.
    - If live fetch fails and a cache exists, cache is used as fallback.
    """

    cache_exists = cache_path is not None and cache_path.exists()
    if cache_exists and not force_refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached = cached.rename(columns=str.lower)
        cached.index.name = "date"
        return cached

    start_dt = _to_datetime(start_date)
    end_dt_inclusive = _to_datetime(end_date)
    end_dt_exclusive = end_dt_inclusive + timedelta(days=1)

    frames: list[pd.DataFrame] = []
    current = start_dt
    while current < end_dt_exclusive:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt_exclusive)
        chunk = yf.download(
            symbol,
            start=current,
            end=chunk_end,
            interval=interval,
            prepost=include_prepost,
            progress=False,
            threads=False,
            multi_level_index=False,
        )

        if not chunk.empty:
            frames.append(_normalize_yfinance_frame(chunk, local_timezone))

        current = chunk_end

    if not frames:
        if fallback_csv_path is not None and fallback_csv_path.exists():
            fallback = pd.read_csv(fallback_csv_path, index_col=0, parse_dates=True)
            fallback = fallback.rename(columns=str.lower)
            fallback.index.name = "date"
            return fallback
        if cache_exists:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached = cached.rename(columns=str.lower)
            cached.index.name = "date"
            return cached
        raise RuntimeError(
            "No data returned from Yahoo Finance and no cache available. "
            "As of 2026, Yahoo restricts older intraday windows. "
            "Provide a local fallback CSV or use a vendor with deep intraday history."
        )

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.loc[(merged.index >= start_dt) & (merged.index < end_dt_exclusive)]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(cache_path)

    return merged


def prepare_after_hours(
    hourly_data: pd.DataFrame,
    start_hour: int = 1,
    end_hour: int = 16,
) -> pd.DataFrame:
    """Filter to the target session window and drop unusable rows."""

    mask = (hourly_data.index.hour >= start_hour) & (hourly_data.index.hour <= end_hour)
    filtered = hourly_data.loc[mask].copy()
    cleaned = filtered.dropna(subset=["open", "high", "low", "close"])
    return cleaned
