# Argento-Lite

Live app: https://argento-lite.streamlit.app/

A rebuild of the portfolio-construction method Plutus describes publicly as Argento (Argento: Risk-Reward
Portfolio Construction, public edition, August 2026), run on 12 public ETFs, plus one idea of my own: what
would a single-company cap cost?

Not investment advice. Public ETFs stand in for Plutus's model portfolios, and every policy constant and ETF
minimum is illustrative or hypothetical.

## What it does

Argento turns an account size and a risk level into a specific mix of funds, at each of 10 risk levels from
calm to wild. This project rebuilds that recipe: estimate returns, measure risk, list the allowed mixes, drop
the unaffordable ones, then pick the best mix at each risk level.

The addition is an optional concentration cap. ETFs overlap (SPY, QQQ, XLK and SMH all hold NVIDIA), so a mix
can put a large share of a client's money into one company, and Argento sets no limit on that. The cap drops
the mixes that break it, rebuilds the frontier from what is left, redoes the ladder and compares level by level.

## Run it

```bash
make setup      # once: creates .venv and installs dependencies
make fetch      # optional, needs internet: refreshes the cached data in data/cache/
make test       # 72 tests
make app        # opens http://localhost:8501
```

The app reads the cached prices and holdings in data/cache/ (committed, about 650 KB) and never touches the
network, so a fresh clone runs after `make setup`.

## Limits

- Free data gives only each ETF's top 10 holdings, so company shares are lower bounds.
- Holdings are today's snapshot, applied to packages chosen from historical returns.
- SMH is trimmed to 2011-12-20, because Yahoo's earlier history belongs to a different fund. This also sets the
  start of the common window (2011-12-21).
- No Income track, no retirement variant, no bootstrap diagnostics.
- The full-history covariance uses the common history of all ETFs, not Argento's ragged-history method.

## Layout

| Path | What it does |
|---|---|
| `config/` | Policy constants, the 12 ETFs with hypothetical minimums, company aliases. |
| `src/argento_lite/data/` | Price and holdings download with a parquet cache. |
| `returns.py`, `risk.py` | Selection return and selection risk. |
| `candidates.py`, `feasibility.py` | The allowed packages and the dollar-minimum check. |
| `frontier.py`, `ladder.py` | Pareto frontier, then one package per risk level. |
| `lookthrough.py` | The extension: company exposure through ETFs, and ETF overlap. |
| `catalog.py` | Calls the above in order. If you read one file, read this one. |
| `report.py`, `plots.py` | Tables and charts for the app. |
| `app/` | Streamlit UI and stylesheet. No selection math. |
| `tests/` | Hand-made examples per rule, an end-to-end run on synthetic prices, and a headless app run. |

Scoring the 15,214 candidate packages happens once in `build_catalog`. Changing the account amount or the cap
only reruns `ladder_for_amount`, so the app responds instantly.
