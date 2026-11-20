"""FinSight paper-only trading lab; NOT a live trading application."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

from finsight.trading import PaperLedger, sma_crossover
from finsight.trading.providers import fetch_paper_reference

st.set_page_config(page_title="FinSight | Paper Trading Desk", page_icon="🧪", layout="wide")
st.caption("FINSIGHT / RESEARCH TERMINAL / PAPER SIMULATOR")
st.title("Paper Trading Desk")
st.error("**PAPER ONLY · NO BROKER CONNECTED.** Manual simulated fills at historical closing references. No real orders, no live quotes, no automation, no investment recommendations.")

if "paper_ledger" not in st.session_state:
    st.session_state.paper_ledger = PaperLedger(Path("artifacts/paper_trading.sqlite3"))
if "paper_order_ready" not in st.session_state:
    st.session_state.paper_order_ready = True
ledger: PaperLedger = st.session_state.paper_ledger

status, cash, positions = st.columns(3)
state = ledger.snapshot()
status.metric("Execution state", "ARMED (PAPER)" if ledger.armed else "PAUSED")
cash.metric("Simulated cash", f"${state['cash_cents'] / 100:,.2f}")
positions.metric("Long positions", len(state["positions"]))

with st.sidebar:
    st.header("Human controls")
    if st.button("⛔ PAUSE / KILL SWITCH", type="primary", use_container_width=True):
        ledger.pause()
        st.rerun()
    acknowledgment = st.text_input("Type PAPER ONLY to enable manual simulated fills", value="", key="paper_ack")
    if st.button("Arm THIS browser session", use_container_width=True):
        if ledger.arm(acknowledgment):
            st.success("Paper simulation enabled in this session. Every fill still requires confirmation.")
        else:
            st.warning("Exact acknowledgement required: PAPER ONLY")
    st.caption("Reopening the app starts PAUSED. Controls cannot enable brokerage orders.")

with st.form("market_fetch"):
    ticker = st.text_input("US-listed equity ticker (USD)", value="NVDA").strip().upper()
    requested = st.form_submit_button("Load historical market reference", type="primary")
if requested:
    try:
        with st.spinner("Loading read-only historical equity data…"):
            st.session_state.paper_reference = fetch_paper_reference(ticker)
    except Exception as exc:
        st.session_state.pop("paper_reference", None)
        st.error(f"Reference unavailable: {exc}")

reference = st.session_state.get("paper_reference")
if reference is not None:
    quote = reference.quote
    st.subheader(f"{quote.ticker} — paper research only")
    st.caption(f"Reference: {quote.source} · {quote.observed_at.isoformat()} · [Original source]({reference.source_url})")
    st.metric("Last completed, unadjusted close", f"${quote.reference_price:,.2f}")
    st.line_chart(reference.adjusted_closes.rename("Adjusted daily closes for strategy"))
    with st.form("strategy"):
        c1, c2 = st.columns(2)
        fast = c1.number_input("Fast SMA sessions", min_value=2, max_value=99, value=10)
        slow = c2.number_input("Slow SMA sessions", min_value=3, max_value=200, value=30)
        investigate = st.form_submit_button("Evaluate closed-candle strategy")
    if investigate:
        try:
            market_today = datetime.now(ZoneInfo("America/New_York")).date()
            signal = sma_crossover(reference.adjusted_closes, fast=int(fast), slow=int(slow), as_of_date=market_today)
            st.info(f"**{signal.action}** · {signal.explanation} · session {signal.as_of_session}; fast/slow SMA {signal.fast_sma:.2f}/{signal.slow_sma:.2f}")
        except ValueError as exc:
            st.warning(f"No signal: {exc}")
    st.divider()
    st.subheader("Manual paper fill")
    st.caption("Strategy suggestions never submit an order. The user must select a side, quantity and type CONFIRM PAPER; the independent risk gate then re-checks everything in a single database transaction.")
    if not st.session_state.paper_order_ready:
        st.warning("Order attempt recorded. Prepare a new order explicitly before submitting again.")
        if st.button("Prepare another PAPER order"):
            st.session_state.paper_order_ready = True
            st.rerun()
    with st.form("paper_order", clear_on_submit=False):
        c1, c2 = st.columns(2)
        side = c1.selectbox("Side", ["BUY", "SELL"])
        shares = c2.number_input("Whole shares", min_value=1, max_value=100, value=1, step=1)
        confirmation = st.text_input("Type CONFIRM PAPER to request a simulated fill")
        submit = st.form_submit_button("Submit PAPER order", type="primary", disabled=not ledger.armed or not st.session_state.paper_order_ready)
    if submit:
        st.session_state.paper_order_ready = False
        key = uuid4().hex  # new attempt only when the human presses submit
        result = ledger.submit(key, quote, side, int(shares), confirmation)
        if result.status == "FILLED_PAPER":
            st.success(f"PAPER fill recorded: {side} {shares} {quote.ticker} at ${quote.reference_price:.2f} historical reference. ID: {key}")
        else:
            st.error(f"Not filled: {result.status}: {result.reason}")

st.divider()
st.subheader("Simulated positions and audit log")
state = ledger.snapshot()
if state["positions"]:
    st.dataframe(pd.DataFrame(state["positions"]), hide_index=True, use_container_width=True)
else:
    st.write("No simulated positions. The account starts with $100,000 virtual cash.")
if state["orders"]:
    st.dataframe(pd.DataFrame(state["orders"]), hide_index=True, use_container_width=True)
st.caption(state["warning"])
st.download_button("Export last 100 audit entries (JSON)", data=__import__("json").dumps(state, indent=2), file_name="finsight_paper_audit.json", mime="application/json")
st.caption("This page does NOT import or control your separate AI-Trading-Desk repository. Its source is not available here. Integration requires a separately reviewed, read-only/paper-only adapter and the existing deterministic broker-evidence safeguards.")
