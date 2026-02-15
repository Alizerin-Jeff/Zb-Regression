"""Session-level metrics and summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def compute_session_metrics(
    after_hours_data: pd.DataFrame,
    *,
    tick_size: float = 1.0 / 32.0,
    ticks_to_close_threshold: int = 4,
) -> pd.DataFrame:
    """Compute per-session deviation/regression metrics."""

    rows: list[dict[str, Any]] = []
    close_threshold = ticks_to_close_threshold * tick_size

    grouped = after_hours_data.groupby(after_hours_data.index.date)
    for session_date, group in grouped:
        if group.empty:
            continue

        close_1am = float(group.iloc[0]["close"])
        max_price = float(group["high"].max())
        min_price = float(group["low"].min())
        max_idx = group["high"].idxmax()
        min_idx = group["low"].idxmin()

        upward_deviation = max_price - close_1am
        downward_deviation = close_1am - min_price

        if upward_deviation > downward_deviation:
            direction = "Upward"
            max_deviation = upward_deviation
            max_hour = int(max_idx.hour)
            start_idx = group.index.get_loc(max_idx)
            post = group.iloc[start_idx:]
            post_extreme = float(post["low"].min())
            regression = max_price - post_extreme
            profitable = post_extreme <= (close_1am + close_threshold)
        else:
            direction = "Downward"
            max_deviation = downward_deviation
            max_hour = int(min_idx.hour)
            start_idx = group.index.get_loc(min_idx)
            post = group.iloc[start_idx:]
            post_extreme = float(post["high"].max())
            regression = post_extreme - min_price
            profitable = post_extreme >= (close_1am - close_threshold)

        regression_ratio = np.nan
        if max_deviation > 0:
            regression_ratio = regression / max_deviation

        rows.append(
            {
                "session_date": pd.Timestamp(session_date),
                "close_1am": close_1am,
                "max_deviation": max_deviation,
                "max_deviation_pct": (max_deviation * 100.0) / close_1am,
                "direction": direction,
                "regression": regression,
                "regression_pct": (regression * 100.0) / close_1am,
                "regression_ratio": regression_ratio,
                "profitable": bool(profitable),
                "max_hour": max_hour,
            }
        )

    return pd.DataFrame(rows)


def summarize_session_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    """Return key aggregate metrics."""

    ratios = metrics["regression_ratio"].dropna()
    return {
        "sessions": float(len(metrics)),
        "avg_max_deviation": float(metrics["max_deviation"].mean()),
        "median_max_deviation": float(metrics["max_deviation"].median()),
        "std_max_deviation": float(metrics["max_deviation"].std()),
        "avg_max_deviation_pct": float(metrics["max_deviation_pct"].mean()),
        "median_max_deviation_pct": float(metrics["max_deviation_pct"].median()),
        "std_max_deviation_pct": float(metrics["max_deviation_pct"].std()),
        "avg_regression_pct": float(metrics["regression_pct"].mean()),
        "median_regression_pct": float(metrics["regression_pct"].median()),
        "std_regression_pct": float(metrics["regression_pct"].std()),
        "median_regression_ratio": float(ratios.median()) if not ratios.empty else np.nan,
        "regression_to_close_frequency": float(metrics["profitable"].mean()),
    }


def regression_survival_curve(
    regression_ratio: pd.Series,
    *,
    max_ratio: float = 2.3,
    step: float = 0.05,
) -> pd.DataFrame:
    """Compute P(ratio >= threshold) across thresholds."""

    clean = regression_ratio.dropna()
    thresholds = np.arange(0.0, max_ratio + step, step)
    probabilities = [(clean >= t).mean() if not clean.empty else np.nan for t in thresholds]
    return pd.DataFrame(
        {
            "ratio_threshold": thresholds,
            "probability_over_threshold": probabilities,
        }
    )


def _smoothstep_inverse(
    x: np.ndarray,
    edge0: float,
    edge1: float,
    a: float = 3.0,
    b: float = 2.0,
) -> np.ndarray:
    """Piecewise smoothstep survival function: ``a*b*t^2 - b*t^3`` in transition region."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return a * t * t * (b - t)


def fit_survival_curve(
    survival: pd.DataFrame,
    *,
    initial_guess: tuple[float, ...] = (1.0, 0.0, 0.3, 4.23),
) -> dict[str, object]:
    """Fit a piecewise smoothstep polynomial to survival curve data.

    Parameters
    ----------
    survival : DataFrame
        Output of :func:`regression_survival_curve` with columns
        ``ratio_threshold`` and ``probability_over_threshold``.
    initial_guess : tuple
        Initial parameters ``(edge0, edge1, a, b)`` for curve_fit.

    Returns
    -------
    dict with ``params``, ``param_names``, ``fitted_values``, ``residuals``, ``sse``.
    """
    x = survival["ratio_threshold"].to_numpy(dtype=float)
    y = survival["probability_over_threshold"].to_numpy(dtype=float)

    popt, _ = curve_fit(_smoothstep_inverse, x, y, p0=initial_guess)
    fitted = _smoothstep_inverse(x, *popt)
    residuals = y - fitted
    sse = float(np.sum(residuals**2))

    return {
        "params": tuple(float(p) for p in popt),
        "param_names": ["edge0", "edge1", "a", "b"],
        "fitted_values": fitted,
        "residuals": residuals,
        "sse": sse,
    }

