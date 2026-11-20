"""Optional FinSight model sentiment. No sentiment-based target-price manipulation."""
from __future__ import annotations

from pathlib import Path

from finsight.inference import load_model, predict_text


def label_articles(articles: list[dict], artifact_path: str = "artifacts/sentiment_model.joblib") -> tuple[list[dict], str]:
    """Label headlines only when a trusted, locally trained artifact is present."""
    model_path = Path(artifact_path)
    if not model_path.is_file():
        return [{**article, "sentiment": None, "positive_score": None} for article in articles], "model_not_trained"
    try:
        model = load_model(model_path)  # NEVER load an untrusted joblib artifact.
    except (FileNotFoundError, ValueError, OSError) as exc:
        return [{**article, "sentiment": None, "positive_score": None} for article in articles], f"model_unavailable: {type(exc).__name__}"
    labelled = []
    for article in articles:
        try:
            outcome = predict_text(model, article["headline"][:2000])
            labelled.append({**article, "sentiment": outcome["label"], "positive_score": outcome["probabilities"].get("positive")})
        except (ValueError, KeyError):
            labelled.append({**article, "sentiment": None, "positive_score": None})
    return labelled, "headline_classification_only"
