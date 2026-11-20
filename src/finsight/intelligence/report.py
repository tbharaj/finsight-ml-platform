"""Portable, source-aware report export."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


def build_report(
    snapshot: Any, diagnostics: dict, distribution: dict | None,
    valuation: list[dict] | None, news: list[dict], news_query_url: str | None,
    warnings: list[str], backtest: dict | None = None,
    *, data_quality: dict | None = None, valuation_inputs: dict | None = None,
) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": snapshot.ticker,
        "company_name": snapshot.company_name,
        "currency": snapshot.currency,
        "financial_currency": snapshot.financial_currency,
        "price_data_as_of": diagnostics["end_date"],
        "market_data_fetched_utc": snapshot.fetched_at_utc,
        "sources": {"market": snapshot.price_source, "news_search": news_query_url, "sec_filing_search": "https://www.sec.gov/edgar/search/"},
        "fundamentals": snapshot.fundamentals,
        "data_quality": data_quality,
        "valuation_inputs": valuation_inputs,
        "historical_price_diagnostics": diagnostics,
        "historical_return_distribution": distribution,
        "earnings_multiple_valuation": valuation,
        "walk_forward_interval_check": backtest,
        "news": news,
        "warnings": warnings,
        "important": "Research prototype only. Earnings multiple values use user-defined assumptions. Historical-return distributions are not calibrated forecasts. News sentiment does not determine price targets.",
    }


def report_json(report: dict) -> str:
    return json.dumps(report, indent=2, default=str, allow_nan=False)
