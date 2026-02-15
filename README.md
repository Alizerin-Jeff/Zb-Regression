# ZB Futures Mean Reversion Analysis

[See the full writeup on Medium](https://medium.com/@jeffreyhartigan/reversion-in-zb-treasury-bond-futures-an-after-hours-analysis-462a4a199bbe)

Quantitative analysis of after-hours mean reversion in 30-year Treasury Bond futures (`ZB=F`). Builds a complete analytical pipeline: deviation/regression metrics, survival curve fitting, distribution validation, an expected value (EV) double-integral framework, and a Nelder-Mead-optimized backtest.

## Key Findings

| Metric | Value |
|--------|-------|
| Sessions analyzed | 505 (2 years) |
| Median regression ratio | 92.5% |
| Regression-to-close frequency | 64.6% |
| Optimized backtest return | 18.2% ($18,201 on $100k) |
| Win rate | 71.3% across 94 trades |

## What This Project Does

- Pulls hourly OHLC futures data via Yahoo Finance (with caching and CSV fallback).
- Computes session-level deviation and regression metrics for after-hours behavior.
- Builds a survival function and fits a smoothstep model for the regression ratio.
- Fits candidate probability distributions and validates against observed moments.
- Constructs an **Expected Value double-integral framework** to analytically evaluate strategy profitability.
- Optimizes strategy parameters with Nelder-Mead, using the EV model to validate the parameter region.
- Backtests a symmetric long/short mean-reversion strategy and compares against EV predictions.

## Methodology

1. **Deviation & regression metrics** — measure how far price moves from the prior close during after-hours trading and how much it retraces by session end.
2. **Survival curve** — fit a piecewise smoothstep to the regression ratio survival function.
3. **Distribution fitting** — fit 8 candidate distributions (LogNormal, Fréchet, Burr, etc.) to regression and deviation data; validate theoretical moments against observed statistics.
4. **EV double integral** — analytically compute expected trade value by integrating over the joint deviation/regression distribution, decomposed into stopped-out, partial-regression, and full-regression components.
5. **Backtest optimization** — Nelder-Mead optimization of entry/stop/take-profit parameters against full historical data.

## Project Structure

```
src/zb_analysis/
├── config.py            # Analysis window, session hours, tick sizing
├── data.py              # Data acquisition + preprocessing (Yahoo/cache/CSV)
├── metrics.py           # Per-session metrics and summary statistics
├── distributions.py     # Distribution fitting, ranking, and validation
├── expected_value.py    # EV double-integral framework and optimization
├── backtest.py          # Strategy backtest + Nelder-Mead optimization
└── __init__.py          # Public API

notebooks/
└── zb_futures_mean_reversion_analysis.ipynb   # Full analysis notebook
```

## Analysis Window

The project is configured for:
- **Period**: `2024-02-13` to `2026-02-13`
- Defined in `src/zb_analysis/config.py`

## Tech Stack

Python 3.13 · scipy · pandas · matplotlib · seaborn · yfinance

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Launch Jupyter and open `notebooks/zb_futures_mean_reversion_analysis.ipynb`.

## Data Note (Yahoo Intraday Limits)

Yahoo Finance limits intraday history to the last ~730 days. The code handles this with a three-step fallback:

1. **Cached file**: `data/processed/zb_hourly_<start>_<end>_dubai.csv`
2. **Live Yahoo fetch**
3. **Fallback CSV**: `data/raw/zb_hourly_<start>_<end>.csv`

If the live fetch fails, place a historical CSV at `data/raw/` with the matching date range. Expected schema:
- Datetime index in the first column
- Columns: `open`, `high`, `low`, `close`
- Optional: `volume`
