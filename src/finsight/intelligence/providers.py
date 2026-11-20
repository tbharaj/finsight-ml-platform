"""Optional third-party data adapters. No API credentials or trading access.

Market prices/metadata: yfinance (unofficial Yahoo interface; research use).
Recent headline links: GDELT DOC API (matching is imperfect, verify sources).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from urllib.parse import urlparse

import pandas as pd

TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,13}$")
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class CompanySnapshot:
    ticker: str
    company_name: str
    currency: str | None
    financial_currency: str | None
    quote_type: str | None
    closes: pd.Series
    fundamentals: dict
    fetched_at_utc: str
    price_source: str


def normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("Invalid ticker: use letters, digits, dots or hyphens (max 14 characters)")
    return ticker


def select_fundamentals(info: dict) -> dict:
    """Whitelisted fields only; never substitute missing financial data with zero."""
    names = (
        "sector", "industry", "country", "website", "marketCap", "totalRevenue",
        "revenueGrowth", "earningsGrowth", "operatingMargins", "profitMargins",
        "freeCashflow", "totalCash", "totalDebt", "trailingEps", "forwardEps",
        "trailingPE", "forwardPE", "sharesOutstanding", "financialCurrency",
    )
    return {name: info.get(name) for name in names}


def fetch_company(ticker: str, period: str = "5y") -> CompanySnapshot:
    symbol = normalize_ticker(ticker)
    if period not in ("3y", "5y", "10y"):
        raise ValueError("Unsupported history range")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install market extras: pip install -e '.[intelligence]'") from exc
    quote = yf.Ticker(symbol)
    history = quote.history(period=period, auto_adjust=True, actions=False)
    if history is None or history.empty or "Close" not in history:
        raise ValueError(f"No historical adjusted closes available for {symbol}")
    closes = pd.to_numeric(history["Close"], errors="coerce").dropna().sort_index()
    if not closes.index.is_unique:
        closes = closes.loc[~closes.index.duplicated(keep="last")]
    if len(closes) < 40 or (closes <= 0).any():
        raise ValueError("Insufficient or invalid history; unable to analyse")
    try:
        info = quote.get_info() or {}
    except Exception:
        info = {}
    quote_type = info.get("quoteType")
    if quote_type and str(quote_type).upper() not in {"EQUITY"}:
        raise ValueError(f"{symbol} is reported as {quote_type}, not an individual equity")
    return CompanySnapshot(
        ticker=symbol,
        company_name=str(info.get("longName") or info.get("shortName") or symbol),
        currency=info.get("currency"),
        financial_currency=info.get("financialCurrency"),
        quote_type=quote_type,
        closes=closes,
        fundamentals=select_fundamentals(info),
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        price_source=f"https://finance.yahoo.com/quote/{symbol}/history/",
    )


def safe_company_search_name(company_name: str) -> str:
    company = str(company_name).strip()
    company = re.sub(r"\b(?:Inc\.?|Ltd\.?|PLC|Corporation|Corp\.?|Co\.?)$", "", company, flags=re.IGNORECASE).strip(" ,.")
    company = re.sub(r"[^a-zA-Z0-9 '&.\-]", " ", company)
    company = " ".join(company.split())
    if len(company) < 4 or len(company) > 80:
        raise ValueError("Company name too short or ambiguous for news search")
    return company


def parse_gdelt_articles(payload: dict, limit: int = 30) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid GDELT news response")
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("Unexpected GDELT articles format")
    seen_urls: set[str] = set()
    output: list[dict] = []
    for item in articles:
        if not isinstance(item, dict):
            continue
        title, url = str(item.get("title") or "").strip(), str(item.get("url") or "").strip()
        parsed = urlparse(url)
        if not title or parsed.scheme not in {"http", "https"} or not parsed.netloc or url in seen_urls:
            continue
        seen_urls.add(url)
        output.append({
            "headline": title[:700],
            "url": url,
            "source": str(item.get("domain") or parsed.netloc),
            "seen_date": str(item.get("seendate") or ""),
            "language": str(item.get("language") or ""),
        })
        if len(output) >= limit:
            break
    return output


def fetch_company_news(company_name: str, days: int = 30, limit: int = 30) -> tuple[list[dict], str]:
    if days not in (7, 30, 60, 90) or not 1 <= limit <= 50:
        raise ValueError("News range or count not supported")
    company = safe_company_search_name(company_name)
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install market extras: pip install -e '.[intelligence]'") from exc
    params = {
        "query": f'"{company}" (earnings OR revenue OR shares OR profit OR guidance)',
        "mode": "artlist",
        "format": "json",
        "timespan": f"{days}d",
        "sort": "datedesc",
        "maxrecords": limit,
    }
    response = requests.get(GDELT_URL, params=params, timeout=12)
    response.raise_for_status()
    return parse_gdelt_articles(response.json(), limit), response.url
