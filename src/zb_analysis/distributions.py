"""Distribution fitting helpers."""

from __future__ import annotations

import warnings
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


_DIST_MAP = {
    "burr": stats.burr,
    "burr12": stats.burr12,
    "gumbel_r": stats.gumbel_r,
    "lognorm": stats.lognorm,
    "weibull_min": stats.weibull_min,
    "weibull_max": stats.weibull_max,
    "gamma": stats.gamma,
    "invweibull": stats.invweibull,
}


def fit_distributions(
    data: pd.Series | np.ndarray,
    *,
    distribution_names: Iterable[str] | None = None,
    bins: int = 25,
) -> pd.DataFrame:
    """Fit candidate scipy distributions and rank by SSE/AIC/BIC."""

    clean = pd.Series(data).dropna().to_numpy(dtype=float)
    if clean.size == 0:
        raise ValueError("Cannot fit distributions to empty data.")

    names = list(distribution_names or _DIST_MAP.keys())
    hist_density, bin_edges = np.histogram(clean, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    results: list[dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            if name not in _DIST_MAP:
                continue

            dist = _DIST_MAP[name]
            try:
                params = dist.fit(clean)
                fitted_pdf = dist.pdf(bin_centers, *params)
                sse = float(np.sum(np.square(hist_density - fitted_pdf)))

                log_pdf = dist.logpdf(clean, *params)
                log_pdf = np.where(np.isfinite(log_pdf), log_pdf, -1e12)
                log_likelihood = float(np.sum(log_pdf))
                k = len(params)
                n = clean.size
                aic = float((2 * k) - (2 * log_likelihood))
                bic = float((k * np.log(n)) - (2 * log_likelihood))

                results.append(
                    {
                        "distribution": name,
                        "params": tuple(float(p) for p in params),
                        "sse": sse,
                        "aic": aic,
                        "bic": bic,
                    }
                )
            except Exception:
                continue

    if not results:
        raise RuntimeError("No distributions could be fit for provided data.")

    ranked = pd.DataFrame(results).sort_values("sse", ascending=True).reset_index(drop=True)
    return ranked


def _get_param_names(count: int) -> list[str]:
    """Return human-readable parameter names based on parameter count."""
    if count == 2:
        return ["loc", "scale"]
    if count == 3:
        return ["shape", "loc", "scale"]
    if count == 4:
        return ["shape2", "shape1", "loc", "scale"]
    return [f"param{i}" for i in range(count)]


def plot_distribution_fit(
    data: np.ndarray | pd.Series,
    distribution_name: str,
    params: tuple[float, ...],
    *,
    bins: int = 20,
    ax_pair: tuple | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> None:
    """Plot histogram + fitted PDF overlay and residuals bar chart.

    Parameters
    ----------
    data : array-like
        Observed data.
    distribution_name : str
        Name of the scipy distribution (must be in ``_DIST_MAP``).
    params : tuple
        Fitted parameters for the distribution.
    bins : int
        Number of histogram bins.
    ax_pair : tuple of (ax1, ax2) or None
        If provided, draw on these axes; otherwise create a new figure.
    figsize : tuple
        Figure size when creating a new figure.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[np.isfinite(clean)]

    dist = _DIST_MAP[distribution_name]
    counts, bin_edges = np.histogram(clean, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]

    pdf = dist.pdf(bin_centers, *params)
    sse = float(np.sum(np.square(counts - pdf)))
    residuals = counts - pdf

    if ax_pair is not None:
        ax1, ax2 = ax_pair
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: histogram + PDF overlay
    ax1.bar(bin_centers, counts, width=bin_width, alpha=0.25, label="Histogram")
    ax1.plot(bin_centers, pdf, color="red", label=f"{distribution_name} PDF")
    param_names = _get_param_names(len(params))
    param_str = "\n                    ".join(
        f"{name}: {p:.2f}" for name, p in zip(param_names, params)
    )
    ax1.annotate(
        f"Parameters => {param_str}\n\n Sum of Squared Error: {sse:.3f}",
        xy=(0.985, 0.9),
        xycoords="axes fraction",
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.2),
    )
    ax1.set_title(f"Fit of {distribution_name.capitalize()} Distribution")
    ax1.set_xlabel("Size of Move (% of Closing Price)")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # Right panel: residuals
    ax2.bar(bin_centers, residuals, width=bin_width, color="blue")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_title(f"Residuals for {distribution_name.capitalize()}")
    ax2.set_xlabel("Data Bins")
    ax2.set_ylabel("Residuals")

    if ax_pair is None:
        plt.tight_layout()
        plt.show()


def plot_all_distribution_fits(
    data: np.ndarray | pd.Series,
    fit_results: pd.DataFrame,
    *,
    bins: int = 20,
    figsize: tuple[float, float] = (14, 6),
) -> None:
    """Plot distribution fit visualizations for every row in *fit_results*.

    Parameters
    ----------
    data : array-like
        The observed data that was fit.
    fit_results : DataFrame
        Output of :func:`fit_distributions` with ``distribution`` and ``params`` columns.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size for each individual plot.
    """
    for _, row in fit_results.iterrows():
        plot_distribution_fit(
            data,
            row["distribution"],
            row["params"],
            bins=bins,
            figsize=figsize,
        )


def validate_distribution_fit(
    distribution_name: str,
    params: tuple[float, ...],
    observed_mean: float,
    observed_median: float,
) -> dict[str, float]:
    """Compare theoretical vs observed mean/median for a fitted distribution.

    Returns a dict with theoretical and observed values plus error percentages.
    """
    dist = _DIST_MAP[distribution_name]
    theo_mean = float(dist.mean(*params))
    theo_median = float(dist.median(*params))

    mean_err = abs(theo_mean - observed_mean) / abs(observed_mean) * 100.0 if observed_mean != 0 else float("nan")
    median_err = abs(theo_median - observed_median) / abs(observed_median) * 100.0 if observed_median != 0 else float("nan")

    return {
        "distribution": distribution_name,
        "theoretical_mean": theo_mean,
        "observed_mean": observed_mean,
        "mean_error_pct": mean_err,
        "theoretical_median": theo_median,
        "observed_median": observed_median,
        "median_error_pct": median_err,
    }

