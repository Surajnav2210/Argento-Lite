"""Daily adjusted close prices from yfinance, cached to parquet."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PRICES_FILE = "prices.parquet"
META_FILE = "prices.meta.json"


class PriceCacheMissing(RuntimeError):
    """Raised when prices are not cached and fetching is not allowed."""


def _request_meta(tickers: list[str], start: date, end: date) -> dict:
    return {"tickers": sorted(tickers), "start": start.isoformat(), "end": end.isoformat()}


def download_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True, progress=False, group_by="column", threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no price data")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    closes = closes.reindex(columns=tickers)
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
    closes.index.name = "date"
    closes.columns.name = None
    missing = [t for t in tickers if closes[t].dropna().empty]
    if missing:
        raise RuntimeError(f"no prices returned for: {', '.join(missing)}")
    return closes.sort_index()


def load_prices(
    tickers: list[str],
    start: date,
    end: date,
    cache_dir: Path,
    refresh: bool = False,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Cached adjusted closes, fetched once. A sidecar JSON records the request, so the same
    request always returns the same numbers even if Yahoo later revises its history."""
    prices_path = cache_dir / PRICES_FILE
    meta_path = cache_dir / META_FILE
    meta = _request_meta(tickers, start, end)

    if not refresh and prices_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == meta:
            return pd.read_parquet(prices_path)[tickers]

    if not allow_fetch:
        raise PriceCacheMissing(
            f"No cached prices for this universe in {cache_dir}. Run `make fetch` first."
        )

    prices = download_prices(tickers, start, end)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(prices_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return prices


def apply_history_starts(prices: pd.DataFrame, starts: dict[str, date]) -> pd.DataFrame:
    """Blank out prices before each ETF's configured ``history_start``."""
    out = prices.copy()
    for ticker, start in starts.items():
        if ticker in out.columns:
            out.loc[out.index < pd.Timestamp(start), ticker] = float("nan")
    return out


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).iloc[1:]
