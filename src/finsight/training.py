"""Train and export the reference model with reproducible evaluation artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from joblib import dump

from finsight.data.splits import LABEL_NAMES, load_splits
from finsight.models.reproducible import fit_and_evaluate


def run(output_dir: str | Path = "artifacts") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, metadata = load_splits()
    model, scores = fit_and_evaluate(splits)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": metadata,
        "evaluation": scores,
        "model_description": "TF-IDF word 1-2grams, class-balanced logistic regression",
        "limitations": "Research benchmark only; not an investment decision system.",
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dump({"pipeline": model, "label_names": LABEL_NAMES}, output / "sentiment_model.joblib")
    return report
