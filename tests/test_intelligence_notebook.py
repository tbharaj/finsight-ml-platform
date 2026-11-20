"""Validate the notebook structure without needing external market APIs."""
from pathlib import Path

import nbformat


def test_company_research_notebook_parses_and_has_no_fake_outputs():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "FinSight_Company_Intelligence.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) >= 10
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "HORIZON_MONTHS" in code and "fetch_company_news" in code
    compile(code, str(path), "exec")
