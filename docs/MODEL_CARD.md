# Model card — FinSight reference baseline

## Intended use

Research-oriented classification of English financial-news headlines into negative (0), neutral (1), and positive (2). This is **sentiment classification only**. It is not a market-return predictor, investment recommendation engine, or suitable system for trading or financial decisions.

## Dataset and evaluation

- Source: [`lmassaron/FinancialPhraseBank`](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank), derived from the Financial PhraseBank dataset introduced by Malo et al. (2014).
- Data licence noted on dataset card: **CC BY-NC-SA 4.0**. The source data are downloaded on demand, not bundled.
- The loader checks sentence/label types, nonempty text, valid class IDs and duplicate-label conflicts.
- If train/validation/test splits exist, they are preserved and decontaminated by normalised exact headline; otherwise a reproducible stratified 70/15/15 split is made with seed 42.
- Exact text overlap is removed in order of priority: test, validation, train. Near-duplicates and semantic paraphrases are **not** detected.
- TF–IDF is fitted on training sentences only. A fixed class-balanced logistic regression baseline is compared with a majority-class dummy model.

## Measured performance

**Not yet verified against the remote benchmark in this revision.** Running `python -m finsight train` creates a machine-readable `artifacts/evaluation.json` containing validation and test accuracy, macro-F1, balanced accuracy and confusion matrices, plus dataset cleaning counts. Do not quote results without running and inspecting this report.

## Limitations

- Financial PhraseBank is a small labelled news benchmark. It does not represent every market, period, instrument, language or editorial context.
- A fixed split can still be optimistic in the presence of related headlines or sources, even with exact-text decontamination.
- Label disagreement and class imbalance can distort reported accuracy; hence macro-F1 and majority-class comparison.
- Estimated class probabilities are **uncalibrated** by default. No threshold or calibration tuning is performed on the test data.
- The logistic regression baseline provides limited insight into economic causality or future returns.
- The original PyTorch and DistilBERT scripts remain experiments and are not covered by benchmark claims for this version.

## Responsible release

There is no broker integration, live order submission, trade automation, user data ingestion or personal financial profiling. Exported joblib artifacts are pickle-based; load only files you have generated or otherwise trust. This API is an unauthenticated local demonstration and must not be deployed publicly without security controls.

## Citations

See [references](REFERENCES.md) for dataset, modelling framework and transformer references.
