import pandas as pd

from finsight.data.splits import prepare_splits
from finsight.inference import predict_text
from finsight.models.reproducible import fit_and_evaluate


def make_frame(prefix):
    words = {0: "loss decline", 1: "meeting update", 2: "growth profit"}
    return pd.DataFrame({
        "sentence": [f"{words[i % 3]} quarterly {prefix} announcement number {i}" for i in range(24)],
        "label": [i % 3 for i in range(24)],
    })


def trained_model():
    splits, _ = prepare_splits({"train": make_frame("train"), "validation": make_frame("validation"), "test": make_frame("test")})
    return fit_and_evaluate(splits)


def test_training_and_metrics():
    model, scores = trained_model()
    assert scores["label_order"] == ["negative", "neutral", "positive"]
    assert 0 <= scores["model"]["test"]["macro_f1"] <= 1
    assert len(scores["model"]["validation"]["confusion_matrix"]) == 3
    assert scores["majority_baseline"]["test"]["n_examples"] == 24
    result = predict_text(model, "profit increased sharply")
    assert result["label"] in ("negative", "neutral", "positive")
    assert abs(sum(result["probabilities"].values()) - 1) < 0.00001
