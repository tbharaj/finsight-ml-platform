"""End-to-end training/export/prediction regression using offline synthetic data."""

import json

import pandas as pd

from finsight import training
from finsight.data.splits import prepare_splits
from finsight.inference import load_model, predict_text


def test_training_exports_model_and_report_with_conflicting_headlines(monkeypatch, tmp_path):
    def frame(prefix):
        return pd.DataFrame({
            "sentence": [f"{prefix} financial market bulletin {i} " + ("profit growth" if i % 3 == 2 else "loss decline" if i % 3 == 0 else "weekly meeting") for i in range(30)],
            "label": [i % 3 for i in range(30)],
        })

    raw = {split: frame(split) for split in ("train", "validation", "test")}
    raw["train"] = pd.concat([
        raw["train"],
        pd.DataFrame({"sentence": ["TRAIN financial market bulletin 0 loss decline"], "label": [1]}),
    ], ignore_index=True)
    splits, meta = prepare_splits(raw)
    assert meta["ambiguous_unique_headlines_excluded"] == 1

    monkeypatch.setattr(training, "load_splits", lambda: (splits, meta))
    report = training.run(tmp_path)
    report_file = tmp_path / "evaluation.json"
    model_file = tmp_path / "sentiment_model.joblib"
    assert report_file.exists() and model_file.exists()
    assert report["source"]["ambiguous_source_rows_excluded_by_split"]["train"] == 2
    assert json.loads(report_file.read_text()) == report
    prediction = predict_text(load_model(model_file), "Company forecasts higher quarterly profits")
    assert prediction["label"] in {"negative", "neutral", "positive"}
