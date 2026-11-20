"""User-assumption P/E scenario calculator, distinct from forecasting returns."""
from __future__ import annotations

import calendar
from dataclasses import asdict, dataclass
from datetime import date
import math


@dataclass(frozen=True)
class EarningsCase:
    name: str
    annual_eps_growth: float
    terminal_pe: float
    future_eps: float
    implied_share_value: float
    change_from_reference: float

    def to_dict(self) -> dict:
        return asdict(self)


def horizon_date(start: date, months: int) -> date:
    if not isinstance(months, int) or not 1 <= months <= 60:
        raise ValueError("Horizon must be between 1 and 60 whole months")
    index = start.year * 12 + (start.month - 1) + months
    year, m_zero = divmod(index, 12)
    month = m_zero + 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def _finite(value: float, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric value") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite numeric value")
    return parsed


def earnings_valuation(
    trailing_eps: float,
    reference_price: float,
    months: int,
    annual_growth: float,
    terminal_pe: float,
    growth_spread: float = 0.10,
    multiple_spread: float = 0.20,
) -> list[EarningsCase]:
    eps = _finite(trailing_eps, "EPS")
    price = _finite(reference_price, "Reference price")
    growth = _finite(annual_growth, "EPS growth")
    multiple = _finite(terminal_pe, "Terminal P/E")
    growth_delta = _finite(growth_spread, "Growth spread")
    multiple_delta = _finite(multiple_spread, "P/E spread")
    if eps <= 0:
        raise ValueError("P/E valuation requires positive trailing earnings per share")
    if price <= 0:
        raise ValueError("Reference price must be positive")
    if not -0.90 <= growth <= 2.0 or not 0 <= growth_delta <= 0.3:
        raise ValueError("Growth assumptions are outside allowed bounds")
    if not 1 <= multiple <= 250 or not 0 <= multiple_delta <= 0.7:
        raise ValueError("Terminal P/E assumptions are outside allowed bounds")
    target = horizon_date(date.today(), months)  # validates months
    del target
    years = months / 12.0
    cases = []
    for name, g, pe in (
        ("Bear", max(-0.95, growth - growth_delta), multiple * (1 - multiple_delta)),
        ("Base", growth, multiple),
        ("Bull", growth + growth_delta, multiple * (1 + multiple_delta)),
    ):
        forward_eps = eps * (1.0 + g) ** years
        value = forward_eps * pe
        cases.append(EarningsCase(name, g, pe, forward_eps, value, value / price - 1.0))
    return cases


def current_multiple(reference_price: float, trailing_eps: float) -> float | None:
    """Returns market-implied P/E when within sensible illustration bounds."""
    try:
        price, eps = _finite(reference_price, "Price"), _finite(trailing_eps, "EPS")
    except ValueError:
        return None
    if price <= 0 or eps <= 0:
        return None
    multiple = price / eps
    return multiple if 1 <= multiple <= 250 else None
