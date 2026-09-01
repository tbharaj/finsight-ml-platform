from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


DATASET_NAME = "lmassaron/FinancialPhraseBank"


def load_data():
    dataset = load_dataset(DATASET_NAME)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    return train_df, val_df


def main():
    train_df, val_df = load_data()

    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(train_df["sentence"])
    X_val = vectorizer.transform(val_df["sentence"])

    model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
    model.fit(X_train, train_df["label"])

    predictions = model.predict(X_val)

    print(f"Accuracy: {accuracy_score(val_df['label'], predictions):.4f}")
    print()
    print(classification_report(val_df["label"], predictions))


if __name__ == "__main__":
    main()
