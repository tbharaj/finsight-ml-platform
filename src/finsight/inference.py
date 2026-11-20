"""Inference helpers shared by the CLI and API."""

from pathlib import Path

from joblib import load

from finsight.data.splits import LABEL_NAMES


def load_model(path: str | Path):
    """Only load model files produced locally by trusted training runs.

    joblib uses pickle; never load an untrusted file.
    """
    artifact = load(Path(path))
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Unexpected model artifact structure")
    return artifact["pipeline"]


def predict_text(model, text: str) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("Provide non-empty text of at most 2000 characters")
    probabilities = model.predict_proba([text])[0]
    classes = [int(label) for label in model.classes_]
    scores = {LABEL_NAMES[label]: round(float(value), 6) for label, value in zip(classes, probabilities)}
    prediction = int(model.predict([text])[0])
    return {"label": LABEL_NAMES[prediction], "probabilities": scores}
