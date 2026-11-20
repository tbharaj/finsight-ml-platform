"""Offline tests; no external data providers or network access required."""
from datetime import date
import numpy as np
import pandas as pd
import pytest

from finsight.intelligence.analytics import historical_distribution, price_diagnostics, rolling_interval_check, validate_closes
from finsight.intelligence.providers import normalize_ticker, parse_gdelt_articles, safe_company_search_name, select_fundamentals
from finsight.intelligence.valuation import current_multiple, earnings_valuation, horizon_date
from finsight.intelligence.report import report_json


def closes(size=900):
    random = np.random.default_rng(33)
    returns = random.normal(0.0004, 0.017, size=size)
    index = pd.bdate_range("2020-01-02", periods=size)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index)


def test_diagnostics_are_finite_and_respect_data_date():
    analysis = price_diagnostics(closes())
    assert analysis.observations == 900
    assert analysis.last_adjusted_close > 0
    assert analysis.end_date == str(closes().index[-1].date())
    assert 0 < analysis.annualised_volatility < 1
    assert -1 < analysis.maximum_drawdown <= 0


def test_distribution_deterministic_and_ordered():
    a = historical_distribution(closes(), 12, simulations=500, seed=4)
    b = historical_distribution(closes(), 12, simulations=500, seed=4)
    assert a == b
    assert 0 < a.p10 <= a.p50 <= a.p90
    assert 0 <= a.fraction_below_start <= 1


def test_distribution_rejects_bad_horizons_and_short_data():
    with pytest.raises(ValueError, match="Horizon"):
        historical_distribution(closes(), 0)
    with pytest.raises(ValueError, match="253"):
        historical_distribution(closes(90), 12)


def test_closes_reject_unsorted_duplicates_negative():
    data = closes()
    assert validate_closes(data).equals(data)
    with pytest.raises(ValueError, match="sorted"):
        validate_closes(data.iloc[::-1])
    data.iloc[3] = -2
    with pytest.raises(ValueError, match="strictly positive"):
        validate_closes(data)


def test_eps_scenarios_trace_assumptions():
    r = earnings_valuation(5.0, 100.0, 12, 0, 20)
    assert [x.name for x in r] == ["Bear", "Base", "Bull"]
    assert r[1].implied_share_value == pytest.approx(100)
    assert r[0].implied_share_value < r[1].implied_share_value < r[2].implied_share_value
    assert r[1].change_from_reference == pytest.approx(0)


def test_negative_earnings_fail_closed():
    assert current_multiple(50, -1) is None
    with pytest.raises(ValueError, match="positive"):
        earnings_valuation(-1, 50, 12, 0.1, 15)


def test_reject_nonfinite_valuation_inputs():
    with pytest.raises(ValueError, match="finite"):
        earnings_valuation(5, 100, 12, np.nan, 20)
    with pytest.raises(ValueError, match="bounds"):
        earnings_valuation(5, 100, 12, 3, 20)


def test_calendar_accurate_month_horizon():
    assert horizon_date(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert horizon_date(date(2026, 12, 31), 1) == date(2027, 1, 31)


def test_walk_forward_returns_honest_insufficient_data():
    assert rolling_interval_check(closes(500), 12)["status"] == "insufficient_history"
    assert rolling_interval_check(closes(900), 12)["status"] in {"descriptive_only", "too_few_windows"}


def test_ticker_validation_and_company_search():
    assert normalize_ticker(" msft ") == "MSFT"
    assert normalize_ticker("brk-b") == "BRK-B"
    with pytest.raises(ValueError):
        normalize_ticker("BAD TICKER")
    assert safe_company_search_name("NVIDIA Corporation") == "NVIDIA"
    with pytest.raises(ValueError):
        safe_company_search_name("A")


def test_gdelt_parser_omits_duplicate_and_malformed_links():
    payload = {"articles": [
        {"title": "Company earnings", "url": "https://publisher.example/story", "domain": "publisher.example", "seendate": "20261120T100000Z"},
        {"title": "Same link", "url": "https://publisher.example/story"},
        {"title": "Untrusted", "url": "javascript:alert(1)"},
        {"title": "Other", "url": "https://example.com/a"},
    ]}
    result = parse_gdelt_articles(payload)
    assert [a["headline"] for a in result] == ["Company earnings", "Other"]


def test_missing_fundamentals_do_not_become_zeros():
    assert select_fundamentals({})["trailingEps"] is None
    assert "arbitrary_secret" not in select_fundamentals({"arbitrary_secret": "x"})


def test_report_serialisation_rejects_nan():
    with pytest.raises(ValueError):
        report_json({"prediction": float("nan")})
