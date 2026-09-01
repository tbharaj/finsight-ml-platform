from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "lmassaron/FinancialPhraseBank"
OUTPUT_PATH = Path("data/processed/train_clean.csv")


def load_data():
    dataset = load_dataset(DATASET_NAME)
    return dataset["train"].to_pandas()


def clean_data(df):
    cleaned = df.drop_duplicates(subset="sentence").copy()
    return cleaned


def save_data(df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


def main():
    df = load_data()
    cleaned = clean_data(df)

    save_data(cleaned)

    print(f"Original rows: {len(df)}")
    print(f"Clean rows: {len(cleaned)}")
    print(f"Removed rows: {len(df) - len(cleaned)}")


if __name__ == "__main__":
    main()
    