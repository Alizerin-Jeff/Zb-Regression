"""Expected value framework for mean-reversion strategy evaluation.

Implements the double-integral EV model from the original analysis:
- Probability that a trade gets stopped out
- Probability that a trade enters within a range
- Full EV computation via ``scipy.integrate.dblquad``
- Nelder-Mead optimization of EV over (entry, stop, take_profit)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import dblquad
from scipy.optimize import minimize

from .distributions import _DIST_MAP


def stop_probability(
    stop_pct: float,
    deviation_dist_name: str,
    deviation_params: tuple[float, ...],
) -> float:
    """Probability that max deviation exceeds the stop level.

    This is simply ``sf(stop_pct)`` for the deviation distribution.
    """
    dist = _DIST_MAP[deviation_dist_name]
    return float(dist.sf(stop_pct, *deviation_params))


def entry_probability(
    entry_pct: float,
    stop_pct: float,
    deviation_dist_name: str,
    deviation_params: tuple[float, ...],
) -> float:
    """Probability that max deviation falls between entry and stop.

    This is ``cdf(stop) - cdf(entry)`` for the deviation distribution.
    """
    dist = _DIST_MAP[deviation_dist_name]
    return float(
        dist.cdf(stop_pct, *deviation_params) - dist.cdf(entry_pct, *deviation_params)
    )


def compute_ev(
    entry: float,
    stop: float,
    take_profit: float,
    regression_dist_name: str,
    regression_params: tuple[float, ...],
    deviation_dist_name: str,
    deviation_params: tuple[float, ...],
) -> float:
    """Compute expected value of the mean-reversion strategy via double integration.

    The EV consists of three components:

    1. **Partial regression** (trade enters, regression doesn't reach take-profit):
       Integrand = PDF_dev(y) * PDF_reg(x) * (entry - y + x)
       over x in [0, y - take_profit], y in [entry, stop]

    2. **Full regression** (trade enters, regression reaches take-profit):
       = (entry - take_profit) * integral of PDF_dev(y) * PDF_reg(x)
       over x in [y - take_profit, inf], y in [entry, stop]

    3. **Stopped out**: sf_dev(stop) * (entry - stop)

    Parameters
    ----------
    entry : float
        Entry threshold (% of close).
    stop : float
        Stop-loss threshold (% of close).
    take_profit : float
        Take-profit threshold (% of close).
    regression_dist_name : str
        Name of the regression distribution (e.g., ``"lognorm"``).
    regression_params : tuple
        Fitted parameters for the regression distribution.
    deviation_dist_name : str
        Name of the deviation distribution (e.g., ``"invweibull"``).
    deviation_params : tuple
        Fitted parameters for the deviation distribution.

    Returns
    -------
    float
        Expected value per trade.
    """
    dev_dist = _DIST_MAP[deviation_dist_name]
    reg_dist = _DIST_MAP[regression_dist_name]

    # Integrand 1: partial regression (trade doesn't reach take-profit)
    def integrand1(x: float, y: float) -> float:
        return (
            dev_dist.pdf(y, *deviation_params)
            * reg_dist.pdf(x, *regression_params)
            * (entry - y + x)
        )

    # Integrand 2: full regression (take-profit reached)
    def integrand2(x: float, y: float) -> float:
        return (
            dev_dist.pdf(y, *deviation_params)
            * reg_dist.pdf(x, *regression_params)
        )

    result1, _ = dblquad(
        integrand1,
        entry,
        stop,
        lambda y: 0,
        lambda y: y - take_profit,
    )

    result2, _ = dblquad(
        integrand2,
        entry,
        stop,
        lambda y: y - take_profit,
        lambda y: np.inf,
    )

    # Stopped-out component
    ev_stopped = float(dev_dist.sf(stop, *deviation_params)) * (entry - stop)

    return result1 + (entry - take_profit) * result2 + ev_stopped


def optimize_ev(
    regression_dist_name: str,
    regression_params: tuple[float, ...],
    deviation_dist_name: str,
    deviation_params: tuple[float, ...],
    *,
    initial_guess: tuple[float, float, float] = (0.7, 2.2, 0.1),
    maxiter: int = 400,
) -> object:
    """Optimize EV over (entry, stop, take_profit) using Nelder-Mead.

    Parameters
    ----------
    regression_dist_name : str
        Name of the regression distribution.
    regression_params : tuple
        Fitted parameters for the regression distribution.
    deviation_dist_name : str
        Name of the deviation distribution.
    deviation_params : tuple
        Fitted parameters for the deviation distribution.
    initial_guess : tuple
        Starting point ``(entry, stop, take_profit)``.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    scipy.optimize.OptimizeResult
    """

    def objective(params: np.ndarray) -> float:
        entry, stop, tp = float(params[0]), float(params[1]), float(params[2])
        # Boundary validation
        if (
            stop > 5
            or stop < entry
            or stop < tp
            or entry < tp
            or entry < 0
            or (entry - tp) < 0.1
            or tp < 0
        ):
            return 10.0
        return -compute_ev(
            entry,
            stop,
            tp,
            regression_dist_name,
            regression_params,
            deviation_dist_name,
            deviation_params,
        )

    result = minimize(
        objective,
        np.array(initial_guess, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
    )
    return result
