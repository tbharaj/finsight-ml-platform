"""Leakage-aware, deterministic scikit-learn baseline and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from finsight.data.splits import LABEL_NAMES, SEED


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=25000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


def evaluate(model: Any, frame: Any) -> dict:
    y_true = frame["label"].to_numpy()
    y_pred = model.predict(frame["sentence"])
    labels = list(LABEL_NAMES)
    return {
        "n_examples": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def fit_and_evaluate(splits: dict) -> tuple[Pipeline, dict]:
    train = splits["train"]
    model = build_model()
    model.fit(train["sentence"], train["label"])
    majority = DummyClassifier(strategy="most_frequent")
    # The dummy classifier must receive a 2-D feature matrix.
    dummy_train = np.zeros((len(train), 1))
    majority.fit(dummy_train, train["label"])
    scores = {"label_order": [LABEL_NAMES[k] for k in LABEL_NAMES], "model": {}, "majority_baseline": {}}
    for name in ("validation", "test"):
        frame = splits[name]
        scores["model"][name] = evaluate(model, frame)
        dummy_frame = np.zeros((len(frame), 1))
        predictions = majority.predict(dummy_frame)
        true = frame["label"].to_numpy()
        scores["majority_baseline"][name] = {
            "n_examples": int(len(frame)),
            "accuracy": float(accuracy_score(true, predictions)),
            "macro_f1": float(f1_score(true, predictions, labels=list(LABEL_NAMES), average="macro", zero_division=0)),
        }
    return model, scores
