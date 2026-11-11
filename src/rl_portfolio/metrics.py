"""Performance metrics on weekly net returns."""

from __future__ import annotations

import numpy as np

PERIODS_PER_YEAR = 52


def cagr(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    return growth ** (PERIODS_PER_YEAR / returns.size) - 1.0


def annualized_volatility(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio; `risk_free` is an annual rate."""
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 2:
        return 0.0
    excess = returns - risk_free / PERIODS_PER_YEAR
    std = excess.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough loss of the wealth path, as a positive fraction."""
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size == 0:
        return 0.0
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.max(1.0 - wealth / peaks))


def summarize(returns: np.ndarray, turnover: np.ndarray | None = None) -> dict[str, float]:
    summary = {
        "cagr": cagr(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "weeks": float(len(returns)),
    }
    if turnover is not None and len(turnover):
        summary["avg_turnover"] = float(np.mean(turnover))
    return summary
