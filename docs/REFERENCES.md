# References and provenance

The following sources support the project methodology and explain the upstream datasets and algorithms. Citations do **not** independently validate this repository's performance.

1. **Financial PhraseBank.** [Hugging Face dataset card: lmassaron/FinancialPhraseBank](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank). Source of headline/label data, label mapping and upstream split configuration. Licence displayed: CC BY-NC-SA 4.0. Accessed 20 November 2026.
2. **Original dataset research.** Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* Journal of the Association for Information Science and Technology, 65(4), 782–796. [Source dataset card citation](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank).
3. **Baseline modelling.** Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830. [Paper](https://jmlr.org/papers/v12/pedregosa11a.html). [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
4. **Dataset split mechanics.** [Hugging Face Datasets — `train_test_split`](https://huggingface.co/docs/datasets/package_reference/main_classes). Stratified splitting with explicit seed when no complete upstream split set exists.
5. **DistilBERT research context.** Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108). Cited to describe the experimental model family in the pre-existing `transformer.py` script; this paper is not evidence that the local experiment succeeded.

**Attribution and licensing:** Project code is MIT licensed. The dataset is separately licensed and is not redistributed in this repository. No commercial usage rights are inferred from the code licence.
