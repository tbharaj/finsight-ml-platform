"""Validate notebook source and ensure no executed results are committed."""

import json
from pathlib import Path


def test_notebook_valid_json_without_outputs():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "FinSight_Financial_Sentiment.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert len(nb["cells"]) >= 15
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            compile("".join(cell["source"]), str(path), "exec")
