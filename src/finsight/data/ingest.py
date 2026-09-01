from datasets import load_dataset

DATASET_NAME = "lmassaron/FinancialPhraseBank"


def load_data():
    dataset = load_dataset(DATASET_NAME)
    df = dataset["train"].to_pandas()
    return df


def main():
    df = load_data()
    print(df.head())
    print(f"Number of rows: {len(df)}")


if __name__ == "__main__":
    main()
