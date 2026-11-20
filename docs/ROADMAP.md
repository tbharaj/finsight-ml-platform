# Development roadmap (not implemented features)

These milestones are proposals and **must not be described as shipped functionality**.

## Next — validate the actual benchmark

- Run the baseline on the upstream dataset in a fresh environment and publish dated output metrics plus dependency versions.
- Record dataset revision/hash and assess train/validation/test duplicate statistics and any near-duplicate risk.
- Add validation-only hyperparameter tuning; hold the test set until model selection is complete.
- Add calibration plots, class-wise error analysis and failure-case examples where data licensing permits.

## Later — extend modelling and evaluation

- Update the original neural and DistilBERT experiments to use shared dataset/split validation, deterministic seeds and comparable evaluations.
- Add a properly benchmarked FinBERT model, *if* it outperforms simpler approaches on a locked test set.
- Build and benchmark a retrieval component (BM25 / vector search) only with an appropriate corpus and retrieval metrics.
- Add reproducible Docker packaging and an authenticated deployment design, where justified by actual users.

## Not in scope

- Claims of predictive market returns or trading profitability.
- Live trading, automatic order execution or handling real brokerage credentials.
- Regulated or production financial advice.
