from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


DATASET_NAME = "lmassaron/FinancialPhraseBank"

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def load_data():
    dataset = load_dataset(DATASET_NAME)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    return train_df, val_df


def main():
    train_df, val_df = load_data()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df["sentence"])
    X_val = vectorizer.transform(val_df["sentence"])

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=2.0
    )

    model.fit(X_train, train_df["label"])

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)

    results = val_df.copy()
    results["prediction"] = predictions
    results["confidence"] = probabilities.max(axis=1)

    errors = results[
        results["label"] != results["prediction"]
    ].copy()

    errors = errors.sort_values(
        "confidence",
        ascending=False
    )

    print(f"Validation examples: {len(results)}")
    print(f"Misclassified examples: {len(errors)}")
    print()

    print("Confusion matrix:")
    print(
        confusion_matrix(
            results["label"],
            results["prediction"]
        )
    )

    print()
    print("Most confident mistakes:")
    print()

    for _, row in errors.head(10).iterrows():
        actual = LABEL_NAMES[row["label"]]
        predicted = LABEL_NAMES[row["prediction"]]

        print(f"Sentence: {row['sentence']}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {row['confidence']:.3f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
    