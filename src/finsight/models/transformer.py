import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DATASET_NAME = "lmassaron/FinancialPhraseBank"
MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 16
MAX_LENGTH = 96
LEARNING_RATE = 0.001
EPOCHS = 5


class FinancialTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.sentences = dataframe["sentence"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.sentences[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        }


def load_data():
    dataset = load_dataset(DATASET_NAME)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    return train_df, val_df


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device
):
    model.train()

    total_loss = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(
    model,
    data_loader,
    device
):
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predicted = outputs.logits.argmax(dim=1)

            predictions.extend(
                predicted.cpu().tolist()
            )

            true_labels.extend(
                batch["labels"].tolist()
            )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro"
    )

    return accuracy, macro_f1


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_df, val_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    # Freeze the pretrained Transformer.
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False

    model = model.to(device)

    train_dataset = FinancialTextDataset(
        train_df,
        tokenizer
    )

    val_dataset = FinancialTextDataset(
        val_df,
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Training examples: {len(train_df)}")
    print()

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        accuracy, macro_f1 = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )


if __name__ == "__main__":
    main()
    