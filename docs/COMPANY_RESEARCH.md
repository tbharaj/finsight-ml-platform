# FinSight Intelligence — Company Research & Valuation

> **Status: research prototype (not a production financial adviser).** The existing FinSight project trains a financial-headline sentiment classifier. This extension adds an interactive, source-aware company research dashboard. It **does not** claim reliable share-price prediction or guaranteed returns.

## Run locally

From the repository root on the review branch:

```bash
source .venv/bin/activate
python -m pip install -e '.[data,api,dev,intelligence]'
python -m streamlit run intelligence_dashboard.py
```

Enter an equity ticker (e.g. `NVDA`), choose the number of months (1–60), select the news lookback and choose **Run company analysis**. Financial and market-data availability varies by listing, and quotes are not real-time. If the saved FinSight sentiment model is missing, the dashboard still displays sourced headlines but does not invent sentiment scores. To create it, run `python -m finsight train --output-dir artifacts` first (see main README).

### What each number means

| Measure | Source or method | What it does **not** mean |
|---|---|---|
| Adjusted closing price | Historical bars via `yfinance` | A live executable quote |
| Financial metrics | Yahoo Finance summary metadata, with varying reporting periods | Verified SEC financial statements or all companies' audited financials |
| Realised volatility | Std. dev. of daily log returns × sqrt(252) | Predicted future volatility |
| P10 / median / P90 | Empirical 5-day block bootstrap of the past 5 years' daily log returns | Calibrated forecast interval, true probabilities or fair value |
| Walk-forward historical coverage | Rolling historical anchors; fit only on prior closes | Proven future interval coverage |
| Bear / Base / Bull valuation | Trailing EPS grown by user-defined annual growth and multiplied by user-defined terminal P/E; scenarios shift those assumptions | Discounted cash flow, analyst consensus, expected returns or automatic recommendation |
| News links | GDELT DOC search for company name with financial keywords | Complete, verified or necessarily relevant news |
| Sentiment | Existing locally trained FinSight classifier, applied only to headline wording | Prediction of a share-price move or trading edge |

## Price-valuation formulas

For a user-chosen horizon `m` months, trailing earnings per share `EPS`, user-set annual EPS growth `g`, and user-set exit multiple `M`:

`implied_value = EPS * (1 + g) ** (m / 12) * M`

Bear and bull cases vary `g` by an explicit growth spread and `M` by a multiplicative P/E spread. Defaults are **illustrative**; in particular 0% growth does not encode any evidence about the company. Changing assumptions changes the displayed value mechanically. **A fair-value estimate needs independent assumptions, industry evidence, detailed cash-flow modelling and sensitivity analysis.** If EPS is negative, unreported or in an incompatible currency, the app does not calculate a P/E target.

The historical-price distribution is computed **separately** from earnings valuation. News sentiment is **not** incorporated into the price equation: treating an unvalidated text label as an expected return would create false confidence.

## Reproducibility and limitations

- Dated market closes, sources, assumptions, data retrieval UTC time and all available output values can be exported to JSON.
- Data retrieval is on demand and cached for 15 minutes per ticker; availability and licensing are controlled by providers.
- `yfinance` is an unofficial interface and may change or throttle access; it is intended for research/personal use. Verify data terms before commercial deployment.
- GDELT indexes news and may show repetitive, irrelevant, missing or incorrectly matched results. The `seen_date` is **not necessarily original publication time**.
- Financial reporting dates can differ across fields. The app does not automatically fetch or reconcile 10-K/10-Q filings. Verify SEC reports and earnings releases before citing financial claims.
- Extrapolation becomes particularly weak for horizons longer than the observed history's stable regime; a 60-month projection does **not** reflect five years of known future data.
- Price histories may include splits/corporate actions; historical distribution based on adjusted prices must not be interpreted as a backtestable trading strategy.
- No API key, brokerage connection, user trades, portfolio positions, scraping of paywalled content or automatic order placement is included.
- Do not deploy publicly without authentication, usage controls, source licensing reviews and production monitoring.

## Sources and references

1. [Yahoo Finance via `yfinance` API documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html) — historical prices and stock metadata. Unofficial; not an authoritative disclosure source.
2. [GDELT Project: DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) — article list search with original links and detection dates.
3. [SEC: EDGAR API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) and [SEC EDGAR filing search](https://www.sec.gov/edgar/search/) — authoritative US regulatory disclosures for independent verification. The current app links to these; it does not claim to ingest them.
4. [scikit-learn: Machine Learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html) — model tooling used in the existing classifier.
5. [Streamlit documentation](https://docs.streamlit.io/get-started/fundamentals/main-concepts) — interactive Python application framework.

## Proposed future upgrades (not delivered)

- SEC companyfacts/filing-reconciliation adapter with a compliant identifiable User-Agent; cash-flow and DCF models with fiscal-period handling.
- Explicit as-of-time news/filing lookups with domain filters and an industry-relative financial scorecard.
- Sector-appropriate valuation models for banks, REITs and companies with negative earnings, rather than applying one P/E model to every stock.
- Independent historical backtesting of predictions with embargoed out-of-sample periods, survivorship-bias controls and interval calibration reports.
- Multi-provider source reconciliation and company-comparison views.
