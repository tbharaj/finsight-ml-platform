from datasets import load_dataset

DATASET_NAME = "lmassaron/FinancialPhraseBank"


def load_data():
    dataset = load_dataset(DATASET_NAME)
    return dataset["train"].to_pandas()


def validate_data(df):
    print("--- DATA VALIDATION ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("Missing values:")
    print(df.isnull().sum())

    duplicates = df.duplicated(subset="sentence").sum()
    print(f"Duplicate sentences: {duplicates}")

    print("Label distribution:")
    print(df["label"].value_counts().sort_index())

    valid = df["label"].isin([0, 1, 2]).all()
    print(f"Labels valid: {valid}")


def main():
    df = load_data()
    validate_data(df)


if __name__ == "__main__":
    main()
