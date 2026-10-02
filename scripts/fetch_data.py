"""Fetch prices and ETF holdings into data/cache/ (needs network once; the app never fetches)."""

from __future__ import annotations

import argparse

from argento_lite.catalog import HOLDINGS_DIR, fetch_universe_prices
from argento_lite.config import CACHE_DIR, load_config
from argento_lite.data.holdings import load_all_holdings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    config = load_config()
    prices = fetch_universe_prices(config, CACHE_DIR, refresh=args.refresh)
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} tickers, "
          f"{prices.index[0].date()} .. {prices.index[-1].date()}")

    snapshots = load_all_holdings(config.universe.tickers, HOLDINGS_DIR, CACHE_DIR, refresh=args.refresh)
    print("Holdings:")
    for ticker, snap in snapshots.items():
        label = "partial" if snap.is_partial else "none" if not snap.holdings else "complete"
        print(f"  {ticker:4s} {snap.source:8s} {len(snap.holdings):3d} rows  "
              f"coverage {snap.coverage:6.1%}  ({label})")


if __name__ == "__main__":
    main()
