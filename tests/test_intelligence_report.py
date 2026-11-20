"""The research export must preserve the assumptions and quality checks."""
import json
from types import SimpleNamespace

from finsight.intelligence.report import build_report, report_json


def test_export_includes_assumptions_and_data_quality():
    snapshot = SimpleNamespace(
        ticker="NVDA", company_name="Test Equity", currency="USD",
        financial_currency="USD", fetched_at_utc="2026-11-20T12:00:00+00:00",
        price_source="https://example.com/price", fundamentals={"trailingEps": 3.0},
    )
    quality = {"valuation_ready": True, "price_age_days": 1, "issues": []}
    assumptions = {"reference_price": 100, "trailing_eps": 3.0,
                   "horizon_months": 12, "eps_growth_assumption_annual": 0.15,
                   "terminal_pe": 25.0}
    report = build_report(snapshot, {"end_date": "2026-11-19"}, None,
                          [{"name": "Base"}], [], "https://api.example.com/news",
                          [], data_quality=quality, valuation_inputs=assumptions)
    exported = json.loads(report_json(report))
    assert exported["valuation_inputs"] == assumptions
    assert exported["data_quality"] == quality
    assert exported["price_data_as_of"] == "2026-11-19"
    assert exported["sources"]["market"] == "https://example.com/price"
    assert exported["earnings_multiple_valuation"] == [{"name": "Base"}]
