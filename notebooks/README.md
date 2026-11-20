# FinSight interactive notebooks

Open **[FinSight: Financial Sentiment](FinSight_Financial_Sentiment.ipynb)** using the repository's virtual environment:

```bash
python -m pip install -e '.[data,notebook]'
python -m jupyter lab notebooks/FinSight_Financial_Sentiment.ipynb
```

Select the project's kernel and choose **Run All Cells**. The notebook trains and evaluates on the real Financial PhraseBank dataset by default (first use requires internet access).

For an *offline smoke test only*, run `export FINSIGHT_NOTEBOOK_SOURCE=synthetic` before launching Jupyter; scores generated from those artificial examples are not research results. Real-data model artifacts are saved to the ignored `artifacts/` directory.

The notebook uses the validated Python modules in `src/finsight`. For full limitations and primary citations, see the [main README](../README.md).
