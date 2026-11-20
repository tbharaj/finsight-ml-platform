"""Closed-candle-only technical proposals; deliberately no order execution."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySignal:
    action: str  # ENTER_LONG, EXIT_LONG, WAIT
    as_of_session: str
    fast_sma: float
    slow_sma: float
    explanation: str


def sma_crossover(closes: pd.Series, *, fast: int = 10, slow: int = 30, as_of_date: date) -> StrategySignal:
    """Only use sessions strictly BEFORE as_of_date (excludes unfinished bars)."""
    if type(fast) is not int or type(slow) is not int or not 2 <= fast < slow <= 200:
        raise ValueError("Require 2 <= fast < slow <= 200")
    if not isinstance(as_of_date, date):
        raise ValueError("Provide a market-local cutoff date")
    if not isinstance(closes, pd.Series) or not isinstance(closes.index, pd.DatetimeIndex):
        raise ValueError("Requires a time-indexed pandas Series of adjusted daily closes")
    if not closes.index.is_monotonic_increasing or not closes.index.is_unique:
        raise ValueError("Session dates must be sorted and unique")
    prior = closes.loc[[stamp.date() < as_of_date for stamp in closes.index]]
    values = pd.to_numeric(prior, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.isna().any() or (values <= 0).any():
        raise ValueError("Incomplete or invalid historical sessions")
    if len(values) < slow + 2:
        raise ValueError("Not enough completed sessions for crossover detection")
    f = values.rolling(fast).mean()
    s = values.rolling(slow).mean()
    prev_up = float(f.iloc[-2]) > float(s.iloc[-2])
    now_up = float(f.iloc[-1]) > float(s.iloc[-1])
    action = "ENTER_LONG" if now_up and not prev_up else "EXIT_LONG" if prev_up and not now_up else "WAIT"
    msg = ("Fast SMA moved above slow SMA" if action == "ENTER_LONG" else
           "Fast SMA moved below slow SMA" if action == "EXIT_LONG" else
           "No completed-session crossover detected")
    return StrategySignal(action, str(values.index[-1].date()), float(f.iloc[-1]), float(s.iloc[-1]), msg + "; observation only, not an order")
