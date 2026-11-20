# FinSight Intelligence — How to access and use it

> **Access:** FinSight is **source code on GitHub**, not a publicly hosted website. To use its interactive interface, you run it locally in Streamlit and open the local address shown in Terminal (normally `http://localhost:8501`). Alternatively, open the Jupyter notebook. You do **not** need the old, unpublished FinSight folder; cloning the repository downloads the code.

## Option A — Interactive company research dashboard (recommended)

### 1. Download the repository on macOS / Linux

Open **Terminal** and paste one line at a time. If you already cloned it earlier, skip `git clone` and start at `cd`.

```bash
git clone https://github.com/tbharaj/finsight-ml-platform.git
cd finsight-ml-platform
git switch main
git pull --ff-only origin main
```

### 2. Install the application (one time)

Use a virtual environment to avoid conflicts with Anaconda or the system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[intelligence,notebook,data,dev]'
```

Confirm that `python -c "import sys; print(sys.executable)"` ends in `finsight-ml-platform/.venv/bin/python`.

### 3. Launch the dashboard

```bash
python -m streamlit run intelligence_dashboard.py
```

Streamlit prints a **Local URL** in Terminal (typically `http://localhost:8501`). Open **that address** in your browser. Keep the Terminal window running while you use the app. To stop it, press **Control+C**.

### 4. Research a company

1. Enter the listed ticker (e.g. `NVDA`, `AAPL` or `MSFT`), rather than the company name.
2. Choose a **1–60-month horizon** and a **7/30/60/90-day news lookback**.
3. Click **Run company analysis**.
4. Inspect **Market & fundamentals** for dated price history and reported financial fields.
5. Review **Price ranges & valuation**. Set your own annual EPS growth rate, terminal P/E and bear/bull spreads. **These are scenario assumptions, not a forecast trained on future prices.**
6. Open **Company news** for source-linked headlines. Optionally train the FinSight classifier (below) to label headline wording.
7. Open **Sources & export** to download a JSON report including dates, quality warnings, provenance and valuation assumptions.

**Worked illustration:** `NVDA`, **12 months**, **30-day news lookback**. Actual results vary with provider data and the day you run it. We publish no fabricated target price or model accuracy.

### 5. Activate sentiment labels (optional)

The dashboard works without a sentiment-model artifact; in that case it displays news links without model-derived labels. To activate the existing classifier, run this in a *separate* Terminal session from the repository root, with `.venv` activated:

```bash
python -m finsight train --output-dir artifacts
```

If real data are downloaded and training finishes successfully, the local file `artifacts/sentiment_model.joblib` will be available. Restart or rerun the dashboard to classify headlines. **Headline positivity does not predict share-price increases.** Never load a `.joblib` model file from an untrusted source.

## Option B — Jupyter notebooks

For company fundamentals, transparent P/E assumptions, historical-price simulations and recent headlines:

```bash
python -m jupyter lab notebooks/FinSight_Company_Intelligence.ipynb
```

For the original text-classification research workflow:

```bash
python -m jupyter lab notebooks/FinSight_Financial_Sentiment.ipynb
```

In JupyterLab, select the virtual environment Python kernel and use **Run → Run All Cells**. The first download of market data, news or the labelled dataset requires internet access.

## Running on subsequent days

```bash
cd ~/finsight-ml-platform
source .venv/bin/activate
python -m streamlit run intelligence_dashboard.py
```

If the project was cloned somewhere else, replace `~/finsight-ml-platform` with its actual location.

## Common troubleshooting

| Symptom | Fix |
|---|---|
| `quote>` in Terminal | A `"` or `'` was left unclosed. Press **Control+C**; then run commands one at a time. |
| `No module named streamlit` / `No module named jupyter` | Activate `.venv`, then rerun `python -m pip install -e '.[intelligence,notebook,data,dev]'`. |
| Thousands of unrelated `pytest` errors / macOS permission errors | You ran tests from `~`. Run `cd ~/finsight-ml-platform` and `python -m pytest -q tests`. |
| Notebook missing | Run `git switch main` and `git pull --ff-only origin main` inside the repository. |
| News unavailable | GDELT may time out or limit queries. Company information and valuation remain separate; no fake headlines are generated. |
| Sentiment unavailable | The local headline classifier was not trained. Train it using the optional command above. |
| Valuation unavailable | The app rejects P/E values with missing/negative EPS, missing or mismatched currencies, an unverified equity type, or a stale/future-dated reference quote. Inspect the quality warnings before retrying. |
| `Connection refused` at `localhost:8501` | The Streamlit process is not running, or it chose a different port. Use the Local URL printed in Terminal. |

## Important interpretation limits

- The application **is not a hosted website** and has no public login or online price API.
- **P/E bear/base/bull valuations** are calculations from *your* earnings-growth and terminal-multiple assumptions; they are not validated price predictions or analyst consensus targets.
- **Historical return simulations** illustrate outcomes under resampled past behaviour, not accurate future probabilities. No training is implied by those simulations.
- **News headlines and sentiment** provide research context, not buy/sell instructions. GDELT dates can reflect when a story was seen rather than first published.
- Financial-summary field periods may differ. Verify fundamentals in official annual/interim filings before citing them.
- Source metadata and quality flags are included in exports. Unverified or missing data should remain missing rather than silently assumed.

Methodology and primary-source references: [Company research methodology](COMPANY_RESEARCH.md). Code licence: [MIT](../LICENSE). Dataset/provider licences are separate.
