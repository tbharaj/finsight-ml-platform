"""Streamlit interface for FinSight Intelligence (research prototype).

Launch: python -m streamlit run intelligence_dashboard.py
Market/news network calls happen only after the user submits a ticker.
"""
from __future__ import annotations

from datetime import date
from math import isfinite

import pandas as pd
import streamlit as st

from finsight.intelligence.analytics import historical_distribution, price_diagnostics, rolling_interval_check
from finsight.intelligence.providers import fetch_company, fetch_company_news
from finsight.intelligence.report import build_report, report_json
from finsight.intelligence.quality import assess_quality
from finsight.intelligence.sentiment import label_articles
from finsight.intelligence.valuation import current_multiple, earnings_valuation, horizon_date

st.set_page_config(page_title="FinSight | Company Intelligence", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1320px; padding-top: 2rem}
h1 {letter-spacing: -.04em; font-size: 2.45rem!important}
h2 {letter-spacing: -.025em}
div[data-testid="stMetric"] {padding: 15px 18px; border: 1px solid #d1d5db44; border-radius: 12px}
div[data-testid="stAlert"] {border-radius: 10px}
</style>""", unsafe_allow_html=True)

st.caption("FINSIGHT   /   RESEARCH TERMINAL")
st.title("Company Intelligence")
st.markdown("**An evidence-led financial research workspace.** Prices, fundamentals, dated headlines and transparent valuation assumptions—without pretending uncertainty disappears.")
st.info("**Research use only.** Not a recommendation, financial advice, or a reliable prediction of future returns. Market prices are delayed historical closes; news may contain irrelevant or duplicated coverage.")
with st.expander("How to use and access FinSight", expanded=False):
    st.markdown(
        "1. Enter an equity ticker such as **NVDA** in the left sidebar.\n"
        "2. Choose the time horizon and recent-news window, then click **Run company analysis**.\n"
        "3. Read the dated financial snapshot and any **data-quality warnings**.\n"
        "4. On the valuation tab, set your **own** earnings-growth and P/E assumptions.\n"
        "5. Read linked news, then export the evidence and assumptions as JSON.\n\n"
        "**Access:** this app runs on your computer. From the project folder run "
        "`python -m streamlit run intelligence_dashboard.py` and open the Local URL "
        "printed in Terminal (usually http://localhost:8501). "
        "See [full setup and troubleshooting instructions](https://github.com/tbharaj/finsight-ml-platform/blob/main/docs/QUICKSTART.md). "
        "No publicly hosted website is available."
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_market(ticker: str):
    return fetch_company(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(company_name: str, days: int):
    return fetch_company_news(company_name, days=days, limit=40)


with st.sidebar:
    st.header("Research settings")
    with st.form("research_params"):
        ticker = st.text_input("Company ticker", value="NVDA", help="For example: NVDA, MSFT, AAPL. Use the listing's Yahoo-style ticker.")
        months = st.slider("Valuation horizon (months)", min_value=1, max_value=60, value=12, step=1)
        news_days = st.selectbox("News lookback", [7, 30, 60, 90], index=1, format_func=lambda x: f"{x} days")
        submitted = st.form_submit_button("Run company analysis", type="primary", use_container_width=True)
    st.caption("Data is only requested when you run an analysis. News requests may be rate-limited or unavailable.")

if submitted:
    st.session_state["company_query"] = {"ticker": ticker, "months": months, "news_days": news_days}
if "company_query" not in st.session_state:
    st.subheader("Choose a ticker to begin")
    st.markdown("The tool will collect up to five years of adjusted closing prices, available summary financial data and recent linked news headlines.")
    st.stop()

params = st.session_state["company_query"]
try:
    with st.spinner("Retrieving company market data…"):
        company = cached_market(params["ticker"])
except Exception as exc:
    st.error(f"Unable to retrieve verified company prices: {exc}")
    st.caption("No price scenario is shown when the market-data source fails. Check the ticker, connection, and any provider restrictions.")
    st.stop()

try:
    diagnostics = price_diagnostics(company.closes)
except ValueError as exc:
    st.error(f"Cannot analyse historical prices: {exc}")
    st.stop()

quality = assess_quality(company, diagnostics)
st.caption(f"{company.ticker}  /  {company.company_name}  ·  Adjusted price as of {diagnostics.end_date}  ·  {company.currency or 'Currency unavailable'}")
st.caption(f"Data audit: {quality.financial_fields_present}/{quality.financial_fields_possible} financial indicators present · Last price {quality.price_age_days} calendar days ago · Equity verified: {'yes' if quality.equity_verified else 'no'}")
if quality.issues:
    with st.expander("Data-quality warnings — read before interpreting valuations", expanded=not quality.valuation_ready):
        for issue in quality.issues:
            st.warning(issue)

st.header(company.company_name)
if company.currency is None:
    st.warning("Market data did not provide a quote currency. All nominal price outputs must be treated with caution.")

last_close = diagnostics.last_adjusted_close
prev_close = float(company.closes.iloc[-2])
change = (last_close / prev_close - 1) * 100
k1, k2, k3, k4 = st.columns(4)
k1.metric("Last adjusted close", f"{last_close:,.2f} {company.currency or ''}", f"{change:+.2f}% vs previous close")
k2.metric("Annualised realised volatility", f"{diagnostics.annualised_volatility:.1%}")
k3.metric("Maximum historical drawdown", f"{diagnostics.maximum_drawdown:.1%}")
k4.metric("Chosen horizon", f"{params['months']} months", f"to {horizon_date(date.fromisoformat(diagnostics.end_date),params['months']).isoformat()}", delta_color="off")

prices_tab, valuation_tab, news_tab, evidence_tab = st.tabs(["Market & fundamentals", "Price ranges & valuation", "Company news", "Sources & export"])

fundamentals = company.fundamentals
with prices_tab:
    st.subheader("Adjusted closing-price history")
    st.line_chart(company.closes.rename("Adjusted close"))
    st.caption("Adjusted historical closes; not intraday or real-time quotes. Past performance does not establish future performance.")
    st.subheader("Financial snapshot")
    def number_label(value, is_percent=False, currency=False):
        if value is None or not isinstance(value, (int, float)) or not isfinite(float(value)):
            return "Not available"
        if is_percent:
            return f"{float(value):+.1%}"
        if currency:
            scaled = float(value)
            if abs(scaled) >= 1e12:
                return f"{scaled/1e12:,.2f}T"
            if abs(scaled) >= 1e9:
                return f"{scaled/1e9:,.2f}B"
            if abs(scaled) >= 1e6:
                return f"{scaled/1e6:,.2f}M"
        return f"{float(value):,.2f}"

    data = [
        ("Market capitalisation", "marketCap", False, True),
        ("Revenue (reported aggregate)", "totalRevenue", False, True),
        ("Revenue growth (provider)", "revenueGrowth", True, False),
        ("Earnings growth (provider)", "earningsGrowth", True, False),
        ("Operating margin", "operatingMargins", True, False),
        ("Profit margin", "profitMargins", True, False),
        ("Free cash flow (provider)", "freeCashflow", False, True),
        ("Total cash", "totalCash", False, True),
        ("Total debt", "totalDebt", False, True),
        ("Trailing EPS", "trailingEps", False, False),
        ("Forward EPS (unverified estimate)", "forwardEps", False, False),
        ("Trailing P/E", "trailingPE", False, False),
        ("Forward P/E", "forwardPE", False, False),
    ]
    st.dataframe(pd.DataFrame([{
        "Metric": label,
        "Reported value": number_label(fundamentals.get(key), is_pct, is_amount),
        "Units": ("quote currency nominal (check financial currency)" if is_amount else "%" if is_pct else "per share / multiple")
    } for label, key, is_pct, is_amount in data]), hide_index=True, use_container_width=True)
    st.caption(f"Sector: {fundamentals.get('sector') or 'unavailable'} · Industry: {fundamentals.get('industry') or 'unavailable'} · Financial currency: {company.financial_currency or 'unavailable'}. Data fields have different reporting periods: verify against primary filings before making decisions.")

warnings: list[str] = list(quality.issues)

distribution = None
walk_forward = None
cases = None
valuation_inputs = None
with valuation_tab:
    st.subheader("1. Historical-return price distribution")
    st.caption("Samples historical daily log returns in five-day blocks and projects them forward. This describes possible outcomes under repeated historical behaviour; it is **not** a probabilistically calibrated forecast or fair value.")
    if diagnostics.observations >= 253:
        result = historical_distribution(company.closes, params["months"], simulations=3000)
        distribution = result.to_dict()
        low, mid, high = st.columns(3)
        low.metric("10th percentile", f"{result.p10:,.2f}")
        mid.metric("Median simulated price", f"{result.p50:,.2f}")
        high.metric("90th percentile", f"{result.p90:,.2f}")
        st.bar_chart(pd.DataFrame({"Price": [result.p10, result.p50, result.p90]}, index=["P10", "P50", "P90"]))
        st.caption(f"{result.simulations:,} simulated outcomes · {result.days} trading days · source observations: {diagnostics.observations:,}. Approx. {result.fraction_below_start:.0%} of sampled outcomes end below the historical reference close. These are model-generated fractions, not reliable real-world probabilities.")
        walk_forward = rolling_interval_check(company.closes, params["months"])
        if walk_forward["status"] == "descriptive_only":
            st.caption(f"Walk-forward historical 80% interval coverage: {walk_forward['coverage_80pct']:.0%} across {walk_forward['windows']} evaluation windows; this diagnostic does not establish future calibration.")
        else:
            st.caption(f"Historical interval backtest: {walk_forward['status'].replace('_',' ')} ({walk_forward['windows']} windows). Insufficient evidence to report coverage.")
    else:
        warnings.append("Insufficient history for empirical future-price distribution.")
        st.warning("At least 253 valid daily closing prices are required for the scenario distribution.")

    st.divider()
    st.subheader("2. Earnings-multiple valuation scenarios")
    st.markdown("Use **your assumptions**, not a hidden AI-generated target. The model compounds trailing earnings per share over the chosen horizon, then applies a terminal P/E multiple. It is illustrative and omits dividends, dilution, buybacks and balance-sheet adjustments.")
    eps = fundamentals.get("trailingEps")
    pe = current_multiple(last_close, eps)
    if pe is None or not quality.valuation_ready:
        st.warning("P/E valuation withheld: positive earnings, verified equity security, recent price and matching reporting/quote currencies are required. Inspect the data-quality warnings above.")
        warnings.append("Earnings-multiple valuation unavailable: positive comparable trailing EPS not verified.")
    else:
        with st.form("assumption_form"):
            a, b, c, d = st.columns(4)
            growth = a.number_input("Annual EPS growth (%)", min_value=-90.0, max_value=200.0, value=0.0, step=2.0, help="Illustrative 0% initial assumption; change only after reviewing company evidence.")
            multiple = b.number_input("Terminal P/E", min_value=1.0, max_value=250.0, value=float(round(pe, 1)), step=1.0)
            spread = c.number_input("Bear/bull growth spread (pp)", min_value=0.0, max_value=30.0, value=10.0, step=2.0)
            pe_spread = d.number_input("P/E spread (%)", min_value=0.0, max_value=70.0, value=20.0, step=5.0)
            st.form_submit_button("Recalculate scenarios", type="primary")
        try:
            cases = [case.to_dict() for case in earnings_valuation(
                eps, last_close, params["months"], growth / 100.0, multiple, spread / 100.0, pe_spread / 100.0
            )]
            valuation_inputs = {
                "reference_price": last_close,
                "price_as_of": diagnostics.end_date,
                "trailing_eps": float(eps),
                "horizon_months": params["months"],
                "eps_growth_assumption_annual": growth / 100.0,
                "terminal_pe": float(multiple),
                "bear_bull_eps_growth_spread_annual": spread / 100.0,
                "bear_bull_terminal_pe_spread": pe_spread / 100.0,
            }
            st.dataframe(pd.DataFrame([{
                "Scenario": c["name"], "EPS growth / year": f"{c['annual_eps_growth']:+.1%}",
                "Terminal P/E": f"{c['terminal_pe']:.1f}×", "Implied EPS": f"{c['future_eps']:.2f}",
                "Implied share valuation": f"{c['implied_share_value']:.2f} {company.currency or ''}",
                "Change vs reference": f"{c['change_from_reference']:+.1%}",
            } for c in cases]), hide_index=True, use_container_width=True)
            st.caption("Bear/Base/Bull are assumption sensitivities, NOT confidence intervals, consensus analyst price targets or automated buy/sell signals.")
        except ValueError as exc:
            st.error(str(exc))

with news_tab:
    st.subheader("Recent linked company headlines")
    st.caption("GDELT DOC indexes news references. Results are based on company-name matching, can include irrelevant stories, and are not independently fact-checked. Headlines are classified only if you have trained a trusted local FinSight model.")
    articles = []
    query_url = None
    try:
        with st.spinner("Searching linked news coverage…"):
            articles, query_url = cached_news(company.company_name, params["news_days"])
    except Exception as exc:
        st.warning(f"News provider unavailable: {exc}. This does not change the modelled valuation or historical price distribution.")
        warnings.append("Recent company-news retrieval was unavailable, so no news-based assessment can be made.")
    annotated, sentiment_status = label_articles(articles)
    if sentiment_status != "headline_classification_only":
        st.info("Headline sentiment is unavailable until the FinSight model is trained. Run `python -m finsight train --output-dir artifacts` locally; headlines can still be reviewed via links.")
    if annotated:
        frame = pd.DataFrame(annotated)
        st.dataframe(frame.rename(columns={"seen_date": "Seen by GDELT", "headline": "Headline", "source": "Publisher", "url": "Article link", "sentiment": "Model sentiment", "positive_score": "Model positive score"}),
                     column_config={"Article link": st.column_config.LinkColumn("Read source", display_text="Open article")},
                     hide_index=True, use_container_width=True)
        st.caption(f"{len(annotated)} linked headlines found; the model labels headline wording, NOT corporate valuation or future returns.")
    else:
        st.write("No matching headlines returned. Try a different company or lookback window; absence of results is not evidence of absence of news.")

with evidence_tab:
    st.subheader("Provenance and limitations")
    st.markdown("[Market-data listing]({}) · [SEC EDGAR primary filings](https://www.sec.gov/edgar/search/) · [GDELT methodology](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)".format(company.price_source))
    st.write(f"Data retrieval time (UTC): {company.fetched_at_utc}")
    st.write(f"Historical prices through: {diagnostics.end_date}")
    st.write(f"Financial currency: {company.financial_currency or 'not supplied'}")
    if warnings:
        for warning in warnings:
            st.warning(warning)
    st.markdown("**Interpretation rules:** `P10/P90` are empirical simulation quantiles, not promised forecast bands. P/E scenarios depend on user assumptions. A market news headline is not independently verified and sentiment does not feed the valuation formula. Insufficient data results in unavailable outputs rather than invented figures.")
    # Construct report from any available analysis; avoids inventing valuation outputs.
    news_value = locals().get("annotated", [])
    url_value = locals().get("query_url")
    report = build_report(company, diagnostics.to_dict(), distribution, cases, news_value, url_value, warnings, walk_forward, data_quality=quality.to_dict(), valuation_inputs=valuation_inputs)
    st.download_button("Download transparent JSON research report", report_json(report), file_name=f"finsight_{company.ticker.lower()}_{diagnostics.end_date}.json", mime="application/json")
