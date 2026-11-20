"""Offline, manual-confirmation-only paper trade ledger.

No brokerage client, order-routing endpoint, AI decision execution, or live
trading capability exists in this module. Arming is an *in-memory* session
state and is never persisted as an enabled capability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path
import re
import sqlite3
from typing import Literal

TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,13}$")
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
CENT = Decimal("0.01")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_cents(value: object) -> int:
    """Convert finite nonnegative money to exact integer cents."""
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Reference price must be a finite positive number") from exc
    if not d.is_finite() or d <= 0 or d > Decimal("1000000"):
        raise ValueError("Reference price must be finite, positive and within limits")
    cents = int((d / CENT).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if cents < 1:
        raise ValueError("Reference price must round to at least one cent")
    return cents


@dataclass(frozen=True)
class PaperQuote:
    ticker: str
    reference_price: float
    currency: str
    instrument_type: str
    observed_at: datetime
    source: str = "historical_daily_close"


@dataclass(frozen=True)
class RiskLimits:
    starting_cash_cents: int = 10_000_000   # $100,000 paper only
    max_shares_per_order: int = 100
    max_order_notional_cents: int = 100_000  # $1,000
    max_position_notional_cents: int = 500_000  # $5,000
    max_gross_book_cents: int = 2_000_000   # $20,000, based on last recorded reference prices
    max_daily_realised_loss_cents: int = 200_000  # $2,000, excludes open-position P&L
    max_quote_age_hours: int = 96


@dataclass(frozen=True)
class OrderResult:
    request_id: str
    status: Literal["FILLED_PAPER", "REJECTED", "PREVIOUSLY_APPLIED", "PREVIOUSLY_REJECTED"]
    reason: str
    cash_after_cents: int


class PaperLedger:
    """Persistent SQLite simulated account with independent, atomic risk checks.

    All orders are synthetic immediate paper fills at *historical reference*
    prices, without broker execution, spreads, fees, dividends or slippage.
    Do not use this as an execution adapter or real broker accounting system.
    """

    def __init__(self, path: str | Path, limits: RiskLimits | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits or RiskLimits()
        self._armed = False
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY CHECK (id=1), cash_cents INTEGER NOT NULL CHECK (cash_cents>=0))")
            conn.execute("CREATE TABLE IF NOT EXISTS positions (ticker TEXT PRIMARY KEY, shares INTEGER NOT NULL CHECK(shares>=0), avg_cost_cents INTEGER NOT NULL, reference_price_cents INTEGER NOT NULL, reference_at TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS orders (request_id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, ticker TEXT NOT NULL, side TEXT NOT NULL, shares INTEGER NOT NULL, reference_price_cents INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL, cash_after_cents INTEGER NOT NULL, realised_pnl_cents INTEGER NOT NULL DEFAULT 0)")
            conn.execute("CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, time_utc TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO account (id, cash_cents) VALUES (1, ?)", (self.limits.starting_cash_cents,))

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            if conn.in_transaction:
                conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self, phrase: str) -> bool:
        """Explicit human action, scoped only to this Python object/session."""
        self._armed = phrase == "PAPER ONLY"
        self._audit("ARM_PAPER" if self._armed else "ARM_DENIED", "Human acknowledgement")
        return self._armed

    def pause(self) -> None:
        self._armed = False
        self._audit("PAUSE", "Human kill-switch")

    def _audit(self, action: str, detail: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO audit (time_utc, action, detail) VALUES (?,?,?)", (utcnow().isoformat(), action, detail))

    def snapshot(self) -> dict:
        with self._connect() as conn:
            cash = int(conn.execute("SELECT cash_cents FROM account WHERE id=1").fetchone()[0])
            positions = [dict(row) for row in conn.execute("SELECT * FROM positions ORDER BY ticker")]
            orders = [dict(row) for row in conn.execute("SELECT * FROM orders ORDER BY created_utc DESC, rowid DESC LIMIT 100")]
            audit = [dict(row) for row in conn.execute("SELECT * FROM audit ORDER BY id DESC LIMIT 100")]
        return {"mode": "PAPER_ONLY", "armed_this_session": self._armed, "cash_cents": cash,
                "positions": positions, "orders": orders, "audit": audit, "risk_limits": asdict(self.limits),
                "warning": "Book values use last recorded reference prices, not executable quotes; open-position unrealised P&L is not covered by daily-loss limits."}

    def _checks(self, conn: sqlite3.Connection, quote: PaperQuote, side: str, shares: int, now: datetime) -> tuple[list[str], int]:
        failures: list[str] = []
        if side not in ("BUY", "SELL"):
            failures.append("INVALID_SIDE")
        if type(shares) is not int or not 1 <= shares <= self.limits.max_shares_per_order:
            failures.append("SHARE_LIMIT")
        ticker = str(quote.ticker).upper()
        if not TICKER.fullmatch(ticker) or ticker != quote.ticker:
            failures.append("INVALID_TICKER")
        if quote.instrument_type != "EQUITY":
            failures.append("EQUITY_NOT_VERIFIED")
        if quote.currency != "USD":
            failures.append("USD_ONLY")
        if quote.source != "historical_daily_close":
            failures.append("UNTRUSTED_REFERENCE_SOURCE")
        if not isinstance(quote.observed_at, datetime) or quote.observed_at.tzinfo is None:
            failures.append("MISSING_TIMEZONE")
        elif quote.observed_at > now or (now - quote.observed_at).total_seconds() > self.limits.max_quote_age_hours * 3600:
            failures.append("STALE_OR_FUTURE_REFERENCE")
        try:
            unit_cents = to_cents(quote.reference_price)
        except ValueError:
            unit_cents = 0
            failures.append("INVALID_REFERENCE_PRICE")
        if failures:
            return failures, unit_cents
        order_notional = unit_cents * shares
        if order_notional > self.limits.max_order_notional_cents:
            failures.append("ORDER_NOTIONAL_LIMIT")
        pos = conn.execute("SELECT * FROM positions WHERE ticker=?", (ticker,)).fetchone()
        held = int(pos["shares"]) if pos is not None else 0
        if side == "SELL":
            if shares > held:
                failures.append("NO_SHORT_SELLING")
            return failures, unit_cents
        cash = int(conn.execute("SELECT cash_cents FROM account WHERE id=1").fetchone()[0])
        if order_notional > cash:
            failures.append("INSUFFICIENT_PAPER_CASH")
        if (held + shares) * unit_cents > self.limits.max_position_notional_cents:
            failures.append("POSITION_NOTIONAL_LIMIT")
        gross = 0
        for row in conn.execute("SELECT * FROM positions"):
            if row["ticker"] == ticker:
                gross += int(row["shares"]) * unit_cents
            else:
                # Cannot approve new exposures against unknown/stale holdings.
                try:
                    marked = datetime.fromisoformat(row["reference_at"])
                    if marked.tzinfo is None or marked > now or (now - marked).total_seconds() > self.limits.max_quote_age_hours * 3600:
                        failures.append("STALE_OTHER_POSITION_MARK")
                except ValueError:
                    failures.append("INVALID_OTHER_POSITION_MARK")
                gross += int(row["shares"]) * int(row["reference_price_cents"])
        if pos is None:
            gross += order_notional
        else:
            gross += order_notional
        if gross > self.limits.max_gross_book_cents:
            failures.append("GROSS_BOOK_LIMIT")
        day = now.date().isoformat()
        realised = int(conn.execute("SELECT COALESCE(SUM(realised_pnl_cents),0) FROM orders WHERE status='FILLED_PAPER' AND side='SELL' AND substr(created_utc,1,10)=?", (day,)).fetchone()[0])
        if realised <= -self.limits.max_daily_realised_loss_cents:
            failures.append("DAILY_REALISED_LOSS_LIMIT")
        return failures, unit_cents

    def preview(self, quote: PaperQuote, side: str, shares: int) -> list[str]:
        if not self._armed:
            return ["PAUSED"]
        now = utcnow()
        with self._connect() as conn:
            failures, _ = self._checks(conn, quote, side, shares, now)
        return failures

    def submit(self, request_id: str, quote: PaperQuote, side: str, shares: int, confirmation: str) -> OrderResult:
        """Atomically record one synthetic fill or rejection (duplicate-safe)."""
        if not REQUEST_ID.fullmatch(str(request_id)):
            raise ValueError("Request ID must be an 8-80 character alphanumeric idempotency key")
        now = utcnow()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute("SELECT status, cash_after_cents FROM orders WHERE request_id=?", (request_id,)).fetchone()
            if previous is not None:
                status = "PREVIOUSLY_APPLIED" if previous["status"] == "FILLED_PAPER" else "PREVIOUSLY_REJECTED"
                return OrderResult(request_id, status, "Idempotency key has already been used", int(previous["cash_after_cents"]))
            cash = int(conn.execute("SELECT cash_cents FROM account WHERE id=1").fetchone()[0])
            failures = []
            if not self._armed:
                failures.append("PAUSED")
            if confirmation != "CONFIRM PAPER":
                failures.append("EXPLICIT_CONFIRMATION_REQUIRED")
            gate, unit_cents = self._checks(conn, quote, side, shares, now)
            failures.extend(gate)
            reason = ", ".join(sorted(set(failures))) or "Manual paper fill at historical reference close"
            status = "REJECTED" if failures else "FILLED_PAPER"
            realised_pnl_cents = 0
            if not failures:
                amount = unit_cents * shares
                existing = conn.execute("SELECT * FROM positions WHERE ticker=?", (quote.ticker,)).fetchone()
                held = int(existing["shares"]) if existing is not None else 0
                if side == "BUY":
                    cash -= amount
                    total_cost = (held * int(existing["avg_cost_cents"]) if existing is not None else 0) + amount
                    new_shares = held + shares
                    avg_cost = (total_cost + new_shares // 2) // new_shares
                    conn.execute("INSERT INTO positions (ticker,shares,avg_cost_cents,reference_price_cents,reference_at) VALUES (?,?,?,?,?) ON CONFLICT(ticker) DO UPDATE SET shares=excluded.shares, avg_cost_cents=excluded.avg_cost_cents, reference_price_cents=excluded.reference_price_cents, reference_at=excluded.reference_at", (quote.ticker, new_shares, avg_cost, unit_cents, quote.observed_at.isoformat()))
                else:
                    cash += amount
                    realised_pnl_cents = amount - shares * int(existing["avg_cost_cents"])
                    remaining = held - shares
                    if remaining:
                        conn.execute("UPDATE positions SET shares=?, reference_price_cents=?, reference_at=? WHERE ticker=?", (remaining, unit_cents, quote.observed_at.isoformat(), quote.ticker))
                    else:
                        conn.execute("DELETE FROM positions WHERE ticker=?", (quote.ticker,))
                conn.execute("UPDATE account SET cash_cents=? WHERE id=1", (cash,))
            conn.execute("INSERT INTO orders (request_id, created_utc, ticker, side, shares, reference_price_cents, status, reason, cash_after_cents, realised_pnl_cents) VALUES (?,?,?,?,?,?,?,?,?,?)", (request_id, now.isoformat(), str(quote.ticker), str(side), int(shares) if type(shares) is int else 0, unit_cents, status, reason, cash, realised_pnl_cents))
            conn.execute("INSERT INTO audit (time_utc, action, detail) VALUES (?,?,?)", (now.isoformat(), status, json.dumps({"request_id": request_id, "reason": reason, "ticker": quote.ticker, "side": side}, sort_keys=True)))
            return OrderResult(request_id, status, reason, cash)
