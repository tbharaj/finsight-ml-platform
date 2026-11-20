"""Quality gates must fail closed when data are uncertain or out of date."""
from datetime import date
from types import SimpleNamespace
import pytest

from finsight.intelligence.quality import assess_quality


def company(**changes):
    params = {
        "currency": "USD", "financial_currency": "USD", "quote_type": "EQUITY",
        "fundamentals": {"trailingEps": 5.5, "totalRevenue": 1e9,
                         "operatingMargins": 0.3, "totalCash": 2e8},
    }
    params.update(changes)
    return SimpleNamespace(**params)


def diagnostic(end_date="2026-11-19"):
    return SimpleNamespace(end_date=end_date)


def test_healthy_market_snapshot_can_be_valued():
    result = assess_quality(company(), diagnostic(), today=date(2026, 11, 20))
    assert result.valuation_ready is True
    assert result.price_age_days == 1
    assert result.financial_fields_present == 4
    assert result.to_dict()["issues"] == []


def test_stale_quotes_block_valuation():
    result = assess_quality(company(), diagnostic("2026-10-01"), today=date(2026, 11, 20))
    assert not result.valuation_ready
    assert not result.price_recent
    assert any("old" in msg for msg in result.issues)


def test_missing_and_mismatched_currencies_fail_closed():
    for quote, reported in [("USD", "GBP"), ("USD", None), (None, "USD"), ("GBp", "GBP")]:
        result = assess_quality(company(currency=quote, financial_currency=reported), diagnostic(),
                                today=date(2026, 11, 20))
        assert not result.valuation_ready
        assert not result.currency_consistent


def test_unknown_security_type_blocks_equity_valuation():
    result = assess_quality(company(quote_type=None), diagnostic(), today=date(2026, 11, 20))
    assert not result.valuation_ready
    assert not result.equity_verified


def test_negative_nan_and_missing_eps_fail_closed():
    for eps in [-3, 0, None, float("nan"), float("inf")]:
        result = assess_quality(company(fundamentals={"trailingEps": eps}), diagnostic(),
                                today=date(2026, 11, 20))
        assert not result.valuation_ready


def test_future_price_warns_and_does_not_pass():
    result = assess_quality(company(), diagnostic("2026-11-21"), today=date(2026, 11, 20))
    assert not result.valuation_ready
    assert result.price_age_days == -1


def test_quality_does_not_invent_missing_fundamentals():
    result = assess_quality(company(fundamentals={}), diagnostic(), today=date(2026, 11, 20))
    assert result.financial_fields_present == 0
    assert result.financial_fields_possible == 8
    assert not result.valuation_ready
    assert any("sparse" in msg for msg in result.issues)


def test_invalid_age_limit_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        assess_quality(company(), diagnostic(), max_price_age_days=-1)
