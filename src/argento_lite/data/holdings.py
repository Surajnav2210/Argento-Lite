"""ETF holdings, looked up in order: parquet cache, yfinance top 10, data/holdings/{TICKER}.csv, none."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from argento_lite.schemas import Holding, HoldingsSnapshot

COLUMNS = ["symbol", "name", "weight", "source"]


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"holdings_{ticker}.parquet"


def _clean(frame: pd.DataFrame) -> list[Holding]:
    out = []
    for symbol, name, weight in frame[["symbol", "name", "weight"]].itertuples(index=False):
        if pd.notna(weight) and weight > 0 and pd.notna(symbol):
            out.append(Holding(symbol=str(symbol).strip(), name=str(name), weight=float(weight)))
    return out


def fetch_yfinance_holdings(ticker: str) -> list[Holding]:
    """yfinance top holdings, or [] if unavailable (never raises)."""
    try:
        import yfinance as yf

        top = yf.Ticker(ticker).funds_data.top_holdings
    except Exception:  # yfinance raises many kinds of errors for funds without holdings
        return []
    if top is None or len(top) == 0:
        return []
    frame = pd.DataFrame({
        "symbol": top.index.astype(str),
        "name": top["Name"].to_numpy(),
        "weight": top["Holding Percent"].to_numpy(),
    })
    return _clean(frame)


def read_csv_holdings(path: Path) -> list[Holding]:
    return _clean(pd.read_csv(path, dtype={"symbol": str}))


def _write_cache(cache_dir: Path, snap: HoldingsSnapshot) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [(h.symbol, h.name, h.weight, snap.source) for h in snap.holdings], columns=COLUMNS)
    frame.to_parquet(_cache_path(cache_dir, snap.ticker))


def _read_cache(cache_dir: Path, ticker: str) -> HoldingsSnapshot | None:
    path = _cache_path(cache_dir, ticker)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    source = frame["source"].iloc[0] if len(frame) else "none"
    return HoldingsSnapshot(ticker=ticker, holdings=_clean(frame), source=source)


def get_holdings(ticker: str, csv_dir: Path, cache_dir: Path, refresh: bool = False,
                 allow_fetch: bool = True) -> HoldingsSnapshot:
    snap = _read_cache(cache_dir, ticker)
    if (snap is None or refresh) and allow_fetch:
        holdings = fetch_yfinance_holdings(ticker)
        snap = HoldingsSnapshot(ticker=ticker, holdings=holdings,
                                source="yfinance" if holdings else "none")
        _write_cache(cache_dir, snap)          # an empty result is cached too: no re-asking Yahoo

    if snap is None or not snap.holdings:
        csv_path = csv_dir / f"{ticker}.csv"
        if csv_path.exists():
            return HoldingsSnapshot(ticker=ticker, holdings=read_csv_holdings(csv_path), source="csv")
    return snap or HoldingsSnapshot(ticker=ticker, holdings=[], source="none")


def load_all_holdings(tickers: list[str], csv_dir: Path, cache_dir: Path, refresh: bool = False,
                      allow_fetch: bool = True) -> dict[str, HoldingsSnapshot]:
    return {t: get_holdings(t, csv_dir, cache_dir, refresh, allow_fetch) for t in tickers}
