"""Read-only US equity reference-close adapter for the local paper simulator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from finsight.intelligence.providers import normalize_ticker
from .core import PaperQuote

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class PaperReference:
    quote: PaperQuote
    adjusted_closes: pd.Series
    source_url: str


def fetch_paper_reference(symbol: str) -> PaperReference:
    """Never returns live quotes; rejects unverified equity type/currency."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install package extra: pip install -e '.[intelligence]'") from exc
    ticker = normalize_ticker(symbol)
    obj = yf.Ticker(ticker)
    info = obj.get_info() or {}
    if str(info.get("quoteType") or "").upper() != "EQUITY":
        raise ValueError("Provider has not verified this ticker as an equity")
    if info.get("currency") != "USD":
        raise ValueError("Paper trading only supports USD equity reference prices")
    # Using unadjusted daily Close as a paper REFERENCE (not an executable fill).
    history = obj.history(period="2y", auto_adjust=False, actions=False)
    if history is None or history.empty or "Close" not in history:
        raise ValueError("No daily market reference available")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Market history must have dated sessions")
    today_ny = datetime.now(NY).date()
    completed = history.loc[[stamp.tz_convert(NY).date() < today_ny if stamp.tzinfo else stamp.date() < today_ny for stamp in history.index]]
    if len(completed) < 32:
        raise ValueError("At least 32 complete sessions required")
    raw = pd.to_numeric(completed["Close"], errors="coerce")
    adjusted = pd.to_numeric(completed["Adj Close"] if "Adj Close" in completed else completed["Close"], errors="coerce")
    if raw.isna().any() or adjusted.isna().any() or (raw <= 0).any() or (adjusted <= 0).any():
        raise ValueError("Incomplete market closing prices")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted market sessions")
    last = raw.index[-1]
    session = last.tz_convert(NY).date() if last.tzinfo else last.date()
    observed_at = datetime.combine(session, time(16, 0), NY).astimezone(timezone.utc)
    return PaperReference(
        quote=PaperQuote(ticker, float(raw.iloc[-1]), "USD", "EQUITY", observed_at),
        adjusted_closes=adjusted,
        source_url=f"https://finance.yahoo.com/quote/{ticker}/history/",
    )
