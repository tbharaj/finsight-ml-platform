# FinSight — Paper Trading Desk

**Status:** standalone, local, research-only **paper simulator**, not a live broker or a release of the separate `AI-Trading-Desk` codebase. All trading starts **PAUSED**. It is not a profit-making bot and has no measured strategy performance.

## Access from GitHub

1. Open the repository [FinSight](https://github.com/tbharaj/finsight-ml-platform).
2. On a branch containing this feature, find **`pages/1_Paper_Trading_Desk.py`** and **`src/finsight/trading/`**. These are the application page and the independent paper ledger. GitHub shows the code but does not run Streamlit.
3. To run the app on your own Mac or Linux computer, clone/download the project and use the commands below. **There is no public, hosted dashboard or brokerage connection.**

From the project root (Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[intelligence,dev]'
python -m streamlit run intelligence_dashboard.py
```

Open the **Local URL** shown by Terminal, normally `http://localhost:8501`. In the Streamlit sidebar, select **Paper Trading Desk**. You can also run `python -m streamlit run pages/1_Paper_Trading_Desk.py` directly, but launching the main app shows both pages.

## Using the paper bot

1. **Default: PAUSED.** Type `PAPER ONLY`, then click **Arm THIS browser session**. Reopening the app creates a new session and starts PAUSED even though simulated positions persist.
2. Enter a supported **US-listed USD equity** ticker such as `AAPL` and click **Load historical market reference**. Market information is read-only via yfinance; a lack of verified equity type/currency or a stale reference blocks fills.
3. Click **Evaluate closed-candle strategy** to generate an `ENTER_LONG`, `EXIT_LONG` or `WAIT` observation from an adjustable simple-moving-average crossover. Today's potentially unfinished daily bar is excluded. This **never creates an order**.
4. For a manual paper fill, choose `BUY` or `SELL` and whole shares, type **`CONFIRM PAPER`**, and press **Submit PAPER order**. The ledger re-checks the risk gate *inside a SQLite write transaction*; an order may be rejected. After each attempt, explicitly choose **Prepare another PAPER order** before a subsequent request.
5. Review **simulated cash, holdings, fills, rejected orders and audit history**; export a JSON snapshot. Press **PAUSE / KILL SWITCH** in the sidebar to stop additional manual fills immediately.

**Data and execution details:** the displayed reference is the last prior completed *unadjusted* daily closing price, not a real-time executable quote. Adjusted closing history is only for strategy observation. Simulated paper fills use reference close exactly; no transaction costs, spreads, borrowing, slippage, dividends, corporate actions, fills or broker account reconciliation are modelled. The account starts with a virtual $100,000 in an on-disk SQLite database at `artifacts/paper_trading.sqlite3`. Back up or remove that file manually if you want to manage the research data; the app does not delete it.

## Safety invariants

- **No broker or trading API:** no Alpaca endpoints, credentials, buy/sell API calls, background loop, order scheduler, or real-money path exists in this feature.
- **Human authority:** defaults PAUSED on every new browser/Python session; an explicit arming phrase and separate per-attempt confirmation are required. Signals are advisory only. The kill switch disarms the current session.
- **Deterministic fail-closed checks:** verified `EQUITY` type, USD quote, explicit source and timezone, strictly positive finite reference prices, age at most 96h, approved sides and whole-share amounts, sufficient cash, no short selling, and position limits.
- **Illustrative portfolio limits:** $1,000 per paper order, $5,000 per ticker, $20,000 gross exposure **valued at last recorded reference prices**, 100 shares per order, and $2,000 daily **realised** loss threshold before further BUYs. These are *not* institutional or live-trading risk controls. Unrealised position losses are not covered, and market moves between reference dates are not reconciled.
- **Audit and idempotency:** unique request IDs, transactional paper cash and position updates, persisted rejected attempts, append-only audit rows. Retrying the same ID cannot create a second paper fill.
- **No lookahead in technical observations:** latest market-local calendar day excluded, minimum complete sessions required, no decision on a partially formed bar.

This is a **practice simulator**. It does not implement the existing project's full write-ahead order-intent protocol, Alpaca broker-evidence reconciliation, AI-agent council, strategy backtesting or production controls. Those features cannot safely be inferred or copied without the original `AI-Trading-Desk` source and review of its separate execution contracts. A future integration must keep its own **PAUSED** default, broker `paper` endpoint, human gates and deterministic firewall; never replace it with this educational SQLite ledger.

## Tests

```bash
python -m pytest -q tests/test_paper_trading.py
python -m pytest -q tests
```

The tests use offline simulated fixtures (no market data downloads or trades) and cover PAUSED gating, explicit confirmations, atomic fills, concurrency/idempotency, stale quotes, equity/currency restrictions, size limits, no shorts, restart safety, and closed-candle strategy analysis.
