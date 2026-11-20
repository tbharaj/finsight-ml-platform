"""Point-in-time data-quality guardrails for exploratory equity valuation.

This is a transparency gate, not an automatic stock recommendation or audit
of upstream provider accuracy. It intentionally fails closed when units,
security type, or price freshness cannot be established.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from typing import Any


KEY_FINANCIALS = (
    "totalRevenue", "revenueGrowth", "operatingMargins", "profitMargins",
    "freeCashflow", "totalCash", "totalDebt", "trailingEps",
)


@dataclass(frozen=True)
class DataQuality:
    price_as_of: str
    price_age_days: int
    price_recent: bool
    quote_currency: str | None
    reporting_currency: str | None
    currency_consistent: bool
    equity_verified: bool
    financial_fields_present: int
    financial_fields_possible: int
    valuation_ready: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["issues"] = list(self.issues)
        return data


def assess_quality(snapshot: Any, diagnostics: Any, today: date | None = None,
                   max_price_age_days: int = 7) -> DataQuality:
    """Explain data sufficiency before presenting numeric P/E valuation.

    The date is taken from the *last observed price bar*, not an API request
    timestamp. A seven-calendar-day freshness gate accommodates weekends but
    may flag longer exchange holidays. The caller can disclose the reason.
    """
    if not isinstance(max_price_age_days, int) or max_price_age_days < 0:
        raise ValueError("max_price_age_days must be a non-negative integer")
    ref_date = date.fromisoformat(str(diagnostics.end_date))
    reference_day = today if today is not None else date.today()
    age = (reference_day - ref_date).days
    price_recent = 0 <= age <= max_price_age_days
    quote = str(snapshot.currency).strip() if snapshot.currency else None
    report = str(snapshot.financial_currency).strip() if snapshot.financial_currency else None
    currency_consistent = bool(
        quote and report and quote.upper() == report.upper()
        and quote not in {"GBp", "GBX", "ZAc"}  # quoted in subunits, not the reporting unit
    )
    equity_verified = str(snapshot.quote_type or "").upper() == "EQUITY"
    fields = snapshot.fundamentals or {}
    populated = sum(
        value is not None
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
        for key in KEY_FINANCIALS
        for value in [fields.get(key)]
    )
    eps = fields.get("trailingEps")
    try:
        positive_eps = eps is not None and not isinstance(eps, bool) and isfinite(float(eps)) and float(eps) > 0
    except (TypeError, ValueError, OverflowError):
        positive_eps = False
    issues: list[str] = []
    if age < 0:
        issues.append("Last price date is in the future relative to the analysis date; verify clock and source.")
    elif not price_recent:
        issues.append(f"Last available price is {age} calendar days old; refresh before valuing.")
    if not currency_consistent:
        issues.append("Quote and reporting currency are missing or different; per-share P/E valuation is blocked.")
    if not equity_verified:
        issues.append("Provider has not confirmed security type EQUITY; equity-specific valuation is blocked.")
    if not positive_eps:
        issues.append("Positive trailing earnings per share are missing or invalid; P/E valuation is unavailable.")
    if populated < 4:
        issues.append("Fewer than four of eight financial indicators were supplied; fundamentals are sparse.")
    return DataQuality(
        price_as_of=str(ref_date), price_age_days=age, price_recent=price_recent,
        quote_currency=quote, reporting_currency=report,
        currency_consistent=currency_consistent, equity_verified=equity_verified,
        financial_fields_present=populated,
        financial_fields_possible=len(KEY_FINANCIALS),
        valuation_ready=bool(price_recent and currency_consistent and equity_verified and positive_eps),
        issues=tuple(issues),
    )
