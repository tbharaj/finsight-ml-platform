from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


DATASET_NAME = "lmassaron/FinancialPhraseBank"


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

    accuracy = accuracy_score(
        val_df["label"],
        predictions
    )

    macro_f1 = f1_score(
        val_df["label"],
        predictions,
        average="macro"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()

    print(
        classification_report(
            val_df["label"],
            predictions
        )
    )


if __name__ == "__main__":
    main()
    