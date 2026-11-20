"""Offline regression tests for a strictly manual, non-broker paper simulator."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

import pandas as pd
import pytest

from finsight.trading import PaperLedger, PaperQuote, RiskLimits, sma_crossover
from finsight.trading.core import to_cents


def quote(price=100.0, *, ticker="AAPL", currency="USD", instrument_type="EQUITY", age_hours=1, source="historical_daily_close"):
    return PaperQuote(ticker, price, currency, instrument_type, datetime.now(timezone.utc) - timedelta(hours=age_hours), source)


@pytest.fixture
def ledger(tmp_path):
    return PaperLedger(tmp_path / "paper.sqlite3")


def key(n):
    return f"test-order-{n:04d}"


def test_starts_paused_and_refuses_unconfirmed_order(ledger):
    assert ledger.armed is False
    result = ledger.submit(key(1), quote(), "BUY", 1, "CONFIRM PAPER")
    assert result.status == "REJECTED" and "PAUSED" in result.reason
    assert ledger.snapshot()["cash_cents"] == 10_000_000
    assert ledger.snapshot()["positions"] == []


def test_only_exact_arm_phrase_and_confirmation(ledger):
    assert not ledger.arm("paper only")
    assert ledger.arm("PAPER ONLY")
    result = ledger.submit(key(1), quote(), "BUY", 1, "not confirmed")
    assert "EXPLICIT_CONFIRMATION_REQUIRED" in result.reason
    assert ledger.snapshot()["positions"] == []
    ledger.pause()
    assert not ledger.armed


def test_successful_paper_buy_sell_and_realised_pnl(ledger):
    ledger.arm("PAPER ONLY")
    buy = ledger.submit(key(1), quote(100), "BUY", 2, "CONFIRM PAPER")
    assert buy.status == "FILLED_PAPER" and buy.cash_after_cents == 10_000_000 - 20_000
    sell = ledger.submit(key(2), quote(110), "SELL", 2, "CONFIRM PAPER")
    assert sell.status == "FILLED_PAPER"
    assert sell.cash_after_cents == 10_000_000 + 2_000
    assert ledger.snapshot()["orders"][0]["realised_pnl_cents"] == 2_000
    assert ledger.snapshot()["positions"] == []


def test_idempotency_filled_and_rejected(ledger):
    ledger.arm("PAPER ONLY")
    request = key(1)
    r1 = ledger.submit(request, quote(100), "BUY", 1, "CONFIRM PAPER")
    r2 = ledger.submit(request, quote(20), "BUY", 10, "CONFIRM PAPER")
    assert r1.status == "FILLED_PAPER" and r2.status == "PREVIOUSLY_APPLIED"
    assert len(ledger.snapshot()["orders"]) == 1
    assert ledger.snapshot()["positions"][0]["shares"] == 1
    denied = ledger.submit(key(2), quote(currency="GBP"), "BUY", 1, "CONFIRM PAPER")
    repeated = ledger.submit(key(2), quote(100), "BUY", 1, "CONFIRM PAPER")
    assert denied.status == "REJECTED" and repeated.status == "PREVIOUSLY_REJECTED"


@pytest.mark.parametrize("bad,expected", [
    (quote(age_hours=200), "STALE_OR_FUTURE_REFERENCE"),
    (quote(age_hours=-1), "STALE_OR_FUTURE_REFERENCE"),
    (quote(currency="GBp"), "USD_ONLY"),
    (quote(instrument_type="ETF"), "EQUITY_NOT_VERIFIED"),
    (quote(source="unknown"), "UNTRUSTED_REFERENCE_SOURCE"),
    (quote(price=float("nan")), "INVALID_REFERENCE_PRICE"),
    (quote(ticker="AAPL;DROP"), "INVALID_TICKER"),
])
def test_fail_closed_reference_types_and_freshness(tmp_path, bad, expected):
    ledger = PaperLedger(tmp_path / "sim.sqlite3")
    ledger.arm("PAPER ONLY")
    r = ledger.submit(key(1), bad, "BUY", 1, "CONFIRM PAPER")
    assert expected in r.reason
    assert r.status == "REJECTED"


def test_order_size_and_position_caps(ledger):
    ledger.arm("PAPER ONLY")
    assert "ORDER_NOTIONAL_LIMIT" in ledger.submit(key(1), quote(100), "BUY", 11, "CONFIRM PAPER").reason
    assert "SHARE_LIMIT" in ledger.submit(key(2), quote(100), "BUY", 101, "CONFIRM PAPER").reason
    assert ledger.submit(key(3), quote(100), "BUY", 10, "CONFIRM PAPER").status == "FILLED_PAPER"
    assert "POSITION_NOTIONAL_LIMIT" in ledger.submit(key(4), quote(100), "BUY", 50, "CONFIRM PAPER").reason


def test_no_naked_shorts(ledger):
    ledger.arm("PAPER ONLY")
    r = ledger.submit(key(1), quote(), "SELL", 1, "CONFIRM PAPER")
    assert r.status == "REJECTED" and "NO_SHORT_SELLING" in r.reason


def test_account_persistent_but_arm_never_persisted(tmp_path):
    path = tmp_path / "paper.sqlite3"
    first = PaperLedger(path)
    first.arm("PAPER ONLY")
    first.submit(key(1), quote(), "BUY", 1, "CONFIRM PAPER")
    new_session = PaperLedger(path)
    assert not new_session.armed
    assert len(new_session.snapshot()["positions"]) == 1
    assert new_session.submit(key(2), quote(), "SELL", 1, "CONFIRM PAPER").status == "REJECTED"


def test_concurrent_duplicate_only_one_fill(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    a, b = PaperLedger(path), PaperLedger(path)
    a.arm("PAPER ONLY")
    b.arm("PAPER ONLY")
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda engine: engine.submit(key(1), quote(), "BUY", 1, "CONFIRM PAPER").status, [a, b]))
    assert sorted(outcomes) == ["FILLED_PAPER", "PREVIOUSLY_APPLIED"]
    assert len(a.snapshot()["orders"]) == 1


def test_invalid_request_ids_and_decimal_money(ledger):
    with pytest.raises(ValueError):
        ledger.submit("x", quote(), "BUY", 1, "CONFIRM PAPER")
    assert to_cents("12.345") == 1235
    with pytest.raises(ValueError):
        to_cents("NaN")
    with pytest.raises(ValueError):
        to_cents("0.0001")


def test_stale_other_holding_blocks_new_exposure(tmp_path):
    p = tmp_path / "sim.sqlite3"
    limits = RiskLimits(max_quote_age_hours=96)
    engine = PaperLedger(p, limits)
    engine.arm("PAPER ONLY")
    assert engine.submit(key(1), quote(100,ticker="AAPL"), "BUY", 1, "CONFIRM PAPER").status == "FILLED_PAPER"
    # Tamper mark to simulate an old holding; a new BUY must fail closed.
    with engine._connect() as conn:
        conn.execute("UPDATE positions SET reference_at=? WHERE ticker='AAPL'", ("2020-01-01T16:00:00+00:00",))
    r = engine.submit(key(2), quote(90,ticker="MSFT"), "BUY", 1, "CONFIRM PAPER")
    assert r.status == "REJECTED" and "STALE_OTHER_POSITION_MARK" in r.reason
    # A SELL of the existing position still reduces risk.
    assert engine.submit(key(3), quote(100,ticker="AAPL"), "SELL", 1, "CONFIRM PAPER").status == "FILLED_PAPER"


def test_strategy_excludes_today_even_if_intraday_bar_is_extreme():
    sessions = pd.bdate_range("2026-01-05", periods=70)
    prices = [100.0] * 67 + [101.0, 110.0, 100000.0]
    daily = pd.Series(prices, index=sessions)
    today = sessions[-1].date()
    first = sma_crossover(daily, fast=3, slow=10, as_of_date=today)
    daily.iloc[-1] = 0.001
    second = sma_crossover(daily, fast=3, slow=10, as_of_date=today)
    assert first == second
    assert first.as_of_session == str(sessions[-2].date())


def test_strategy_errors_on_unsorted_or_insufficient_data():
    sessions = pd.bdate_range("2025-01-01", periods=10)
    with pytest.raises(ValueError, match="Not enough"):
        sma_crossover(pd.Series([100.0] * 10, index=sessions), as_of_date=date(2025,2,1))
    with pytest.raises(ValueError, match="sorted"):
        sma_crossover(pd.Series([100.0] * 10, index=sessions[::-1]), as_of_date=date(2025,2,1))
