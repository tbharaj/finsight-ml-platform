"""Deterministic historical-price diagnostics and empirical scenario sampling.

Sampling from historical returns is NOT a validated share-price forecasting model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PriceDiagnostics:
    observations: int
    start_date: str
    end_date: str
    last_adjusted_close: float
    annualised_log_return: float
    annualised_volatility: float
    maximum_drawdown: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalDistribution:
    months: int
    days: int
    simulations: int
    p10: float
    p50: float
    p90: float
    fraction_below_start: float
    method: str = "Five-day circular block bootstrap of historical daily log returns"

    def to_dict(self) -> dict:
        return asdict(self)


def validate_closes(closes: pd.Series, minimum: int = 2) -> pd.Series:
    if not isinstance(closes, pd.Series):
        raise TypeError("Historical closing prices must be a pandas Series")
    numeric = pd.to_numeric(closes, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(numeric) < minimum:
        raise ValueError(f"At least {minimum} valid daily closes are required")
    if (numeric <= 0).any():
        raise ValueError("Prices must be strictly positive")
    if not numeric.index.is_unique or not numeric.index.is_monotonic_increasing:
        raise ValueError("Historical dates must be unique and sorted ascending")
    return numeric.astype(float)


def price_diagnostics(closes: pd.Series) -> PriceDiagnostics:
    prices = validate_closes(closes, minimum=20)
    daily_log = np.diff(np.log(prices.to_numpy()))
    running_high = prices.cummax()
    return PriceDiagnostics(
        observations=len(prices),
        start_date=str(pd.Timestamp(prices.index[0]).date()),
        end_date=str(pd.Timestamp(prices.index[-1]).date()),
        last_adjusted_close=float(prices.iloc[-1]),
        annualised_log_return=float(np.mean(daily_log) * DAYS_PER_YEAR),
        annualised_volatility=float(np.std(daily_log, ddof=1) * sqrt(DAYS_PER_YEAR)),
        maximum_drawdown=float((prices / running_high - 1.0).min()),
    )


def historical_distribution(
    closes: pd.Series,
    months: int,
    simulations: int = 3000,
    seed: int = 42,
    block_size: int = 5,
) -> HistoricalDistribution:
    if not isinstance(months, int) or not 1 <= months <= 60:
        raise ValueError("Horizon must be between 1 and 60 whole months")
    if not 100 <= simulations <= 10000:
        raise ValueError("Simulation count must be between 100 and 10,000")
    if not 1 <= block_size <= 21:
        raise ValueError("Block size must be between 1 and 21 trading days")
    prices = validate_closes(closes, minimum=253)
    returns = np.diff(np.log(prices.to_numpy()))
    forecast_days = max(1, round(DAYS_PER_YEAR * months / 12))
    block_count = (forecast_days + block_size - 1) // block_size
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(returns), size=(simulations, block_count))
    indices = (starts[:, :, None] + np.arange(block_size)) % len(returns)
    cumulative = returns[indices].reshape(simulations, -1)[:, :forecast_days].sum(axis=1)
    simulated_prices = prices.iloc[-1] * np.exp(np.clip(cumulative, -20, 20))
    low, middle, high = np.quantile(simulated_prices, [0.10, 0.50, 0.90])
    return HistoricalDistribution(
        months=months,
        days=forecast_days,
        simulations=simulations,
        p10=float(low),
        p50=float(middle),
        p90=float(high),
        fraction_below_start=float(np.mean(simulated_prices < prices.iloc[-1])),
    )


def rolling_interval_check(
    closes: pd.Series,
    months: int,
    lookback_days: int = 504,
    simulations: int = 250,
) -> dict:
    """Walk-forward interval coverage; no future prices enter any training window.

    Anchors are spaced by the projection horizon to reduce overlap.
    Do not interpret historical coverage as proof of forward calibration.
    """
    prices = validate_closes(closes, minimum=253)
    n_days = max(1, round(DAYS_PER_YEAR * months / 12))
    if len(prices) < lookback_days + n_days + 1:
        return {"status": "insufficient_history", "windows": 0, "coverage_80pct": None}
    anchor_indices = range(lookback_days - 1, len(prices) - n_days, max(21, n_days))
    observations = []
    for i, anchor in enumerate(anchor_indices):
        sample = prices.iloc[anchor - lookback_days + 1: anchor + 1]
        forecast = historical_distribution(sample, months, simulations=simulations, seed=200 + i)
        actual = float(prices.iloc[anchor + n_days])
        observations.append((forecast.p10 <= actual <= forecast.p90, abs(forecast.p50 - actual) / actual))
    if len(observations) < 5:
        return {"status": "too_few_windows", "windows": len(observations), "coverage_80pct": None}
    return {
        "status": "descriptive_only",
        "windows": len(observations),
        "coverage_80pct": float(np.mean([entry[0] for entry in observations])),
        "median_absolute_percentage_error": float(np.median([entry[1] for entry in observations])),
        "note": "Overlapping returns and regime changes limit inference; coverage is historical only.",
    }
