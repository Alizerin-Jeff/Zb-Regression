"""Core package for ZB mean reversion analysis."""

from .backtest import BacktestParams, optimize_backtest, run_backtest, summarize_backtest
from .config import AnalysisConfig
from .data import fetch_hourly_data, prepare_after_hours
from .distributions import (
    fit_distributions,
    plot_all_distribution_fits,
    plot_distribution_fit,
    validate_distribution_fit,
)
from .expected_value import compute_ev, entry_probability, optimize_ev, stop_probability
from .metrics import (
    compute_session_metrics,
    fit_survival_curve,
    regression_survival_curve,
    summarize_session_metrics,
)

__all__ = [
    "AnalysisConfig",
    "BacktestParams",
    "compute_ev",
    "compute_session_metrics",
    "entry_probability",
    "fetch_hourly_data",
    "fit_distributions",
    "fit_survival_curve",
    "optimize_backtest",
    "optimize_ev",
    "plot_all_distribution_fits",
    "plot_distribution_fit",
    "prepare_after_hours",
    "regression_survival_curve",
    "run_backtest",
    "stop_probability",
    "summarize_backtest",
    "summarize_session_metrics",
    "validate_distribution_fit",
]
