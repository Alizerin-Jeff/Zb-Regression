"""Backtest and parameter optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class BacktestParams:
    """Backtest parameters as percentages of close."""

    entry_pct: float
    stop_pct: float
    take_profit_pct: float

    def is_valid(self) -> bool:
        if self.entry_pct < 0 or self.take_profit_pct < 0:
            return False
        if self.stop_pct < self.entry_pct:
            return False
        if self.entry_pct < self.take_profit_pct:
            return False
        return True


def run_backtest(
    after_hours_data: pd.DataFrame,
    params: BacktestParams,
    *,
    starting_balance: float = 100_000.0,
    ticks_per_point: int = 32,
    dollars_per_tick: float = 31.25,
    contracts: int = 1,
    max_sessions: Optional[int] = None,
) -> pd.DataFrame:
    """Run symmetric long/short mean-reversion backtest over grouped sessions."""

    if not params.is_valid():
        raise ValueError(f"Invalid backtest parameters: {params}")

    entry_mult = 1 + params.entry_pct / 100.0
    stop_mult = 1 + params.stop_pct / 100.0
    take_profit_mult = 1 + params.take_profit_pct / 100.0

    grouped = after_hours_data.groupby(after_hours_data.index.date)
    balance = starting_balance
    records: list[dict[str, object]] = []

    for i, (session_date, group) in enumerate(grouped):
        if max_sessions is not None and i >= max_sessions:
            break
        if group.empty:
            continue

        one_am_close = float(group.iloc[0]["close"])
        end_price = float(group.iloc[-1]["close"])

        short_entry = one_am_close * entry_mult
        short_stop = one_am_close * stop_mult
        short_tp = one_am_close * take_profit_mult

        long_entry = one_am_close * (2 - entry_mult)
        long_stop = one_am_close * (2 - stop_mult)
        long_tp = one_am_close * (2 - take_profit_mult)

        pnl_short_loss = (short_entry - short_stop) * ticks_per_point * dollars_per_tick * contracts
        pnl_short_win = (short_entry - short_tp) * ticks_per_point * dollars_per_tick * contracts
        pnl_long_loss = (long_stop - long_entry) * ticks_per_point * dollars_per_tick * contracts
        pnl_long_win = (long_tp - long_entry) * ticks_per_point * dollars_per_tick * contracts

        max_price = float(group["high"].max())
        max_idx = group["high"].idxmax()
        min_price = float(group["low"].min())
        min_idx = group["low"].idxmin()

        # Short side
        if max_price >= short_stop:
            balance += pnl_short_loss
            records.append(
                {"date": pd.Timestamp(session_date), "side": "short", "pnl": pnl_short_loss, "balance": balance}
            )
        elif max_price >= short_entry:
            start = group.index.get_loc(max_idx)
            post = group.iloc[start:]
            min_post = float(post["low"].min())
            if min_post <= short_tp:
                pnl = pnl_short_win
            else:
                pnl = (short_entry - end_price) * ticks_per_point * dollars_per_tick * contracts
            balance += pnl
            records.append({"date": pd.Timestamp(session_date), "side": "short", "pnl": pnl, "balance": balance})

        # Long side
        if min_price <= long_stop:
            balance += pnl_long_loss
            records.append({"date": pd.Timestamp(session_date), "side": "long", "pnl": pnl_long_loss, "balance": balance})
        elif min_price <= long_entry:
            start = group.index.get_loc(min_idx)
            post = group.iloc[start:]
            max_post = float(post["high"].max())
            if max_post >= long_tp:
                pnl = pnl_long_win
            else:
                pnl = (end_price - long_entry) * ticks_per_point * dollars_per_tick * contracts
            balance += pnl
            records.append({"date": pd.Timestamp(session_date), "side": "long", "pnl": pnl, "balance": balance})

    return pd.DataFrame(records)


def summarize_backtest(trades: pd.DataFrame, *, starting_balance: float = 100_000.0) -> dict[str, float]:
    """Compute backtest summary statistics."""

    if trades.empty:
        return {
            "starting_balance": starting_balance,
            "ending_balance": starting_balance,
            "total_pnl": 0.0,
            "num_trades": 0.0,
            "win_rate": np.nan,
            "avg_pnl_per_trade": np.nan,
            "max_drawdown": 0.0,
            "return_pct": 0.0,
        }

    pnl = trades["pnl"].astype(float)
    equity = trades["balance"].astype(float)
    peaks = equity.cummax()
    drawdowns = peaks - equity

    return {
        "starting_balance": starting_balance,
        "ending_balance": float(equity.iloc[-1]),
        "total_pnl": float(equity.iloc[-1] - starting_balance),
        "num_trades": float(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl_per_trade": float(pnl.mean()),
        "max_drawdown": float(drawdowns.max()),
        "return_pct": float((equity.iloc[-1] - starting_balance) / starting_balance * 100.0),
    }


def optimize_backtest(
    after_hours_data: pd.DataFrame,
    *,
    initial_guess: tuple[float, float, float] = (0.7, 2.2, 0.1),
    starting_balance: float = 100_000.0,
    maxiter: int = 400,
) -> object:
    """Optimize backtest parameters to maximize ending balance."""

    def objective(values: np.ndarray) -> float:
        p = BacktestParams(float(values[0]), float(values[1]), float(values[2]))
        if not p.is_valid():
            return 1e9
        trades = run_backtest(after_hours_data, p, starting_balance=starting_balance)
        if trades.empty:
            return 1e6
        return -float(trades["balance"].iloc[-1])

    result = minimize(
        objective,
        np.array(initial_guess, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-2},
    )
    return result

