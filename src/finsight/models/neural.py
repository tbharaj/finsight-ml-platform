import copy
import re
from collections import Counter

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


DATASET_NAME = "lmassaron/FinancialPhraseBank"

BATCH_SIZE = 32
EMBEDDING_DIM = 128
LEARNING_RATE = 0.001
MAX_EPOCHS = 20
PATIENCE = 4


def load_data():
    dataset = load_dataset(DATASET_NAME)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    return train_df, val_df


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(sentences, min_freq=2):
    counter = Counter()

    for sentence in sentences:
        counter.update(tokenize(sentence))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode(text, vocab):
    tokens = tokenize(text)

    token_ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    if not token_ids:
        token_ids = [vocab["<UNK>"]]

    return token_ids


class FinancialSentimentDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.sentences = dataframe["sentence"].tolist()
        self.labels = dataframe["label"].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        token_ids = encode(
            self.sentences[index],
            self.vocab
        )

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long)
        )


def collate_batch(batch):
    sentences, labels = zip(*batch)

    padded_sentences = pad_sequence(
        sentences,
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack(labels)

    return padded_sentences, labels


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_classes=3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):
        embedded = self.embedding(x)

        mask = (x != 0).unsqueeze(-1).float()

        embedded = embedded * mask

        summed = embedded.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)

        pooled = summed / lengths

        return self.classifier(pooled)


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    loss_function,
    device
):
    model.train()

    total_loss = 0

    for sentences, labels in data_loader:
        sentences = sentences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(sentences)

        loss = loss_function(
            outputs,
            labels
        )

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
        for sentences, labels in data_loader:
            sentences = sentences.to(device)

            outputs = model(sentences)

            predicted_labels = outputs.argmax(dim=1)

            predictions.extend(
                predicted_labels.cpu().tolist()
            )

            true_labels.extend(
                labels.tolist()
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

    return (
        accuracy,
        macro_f1,
        true_labels,
        predictions
    )


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_df, val_df = load_data()

    vocab = build_vocab(
        train_df["sentence"]
    )

    train_dataset = FinancialSentimentDataset(
        train_df,
        vocab
    )

    val_dataset = FinancialSentimentDataset(
        val_df,
        vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )

    model = SentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    label_counts = torch.bincount(
        torch.tensor(
            train_df["label"].tolist()
        )
    ).float()

    class_weights = (
        len(train_df)
        / (3 * label_counts)
    ).to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_macro_f1 = 0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Vocabulary size: {len(vocab)}")
    print()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device
        )

        accuracy, macro_f1, _, _ = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print()
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)

    accuracy, macro_f1, labels, predictions = evaluate(
        model,
        val_loader,
        device
    )

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best accuracy: {accuracy:.4f}")
    print(f"Best Macro F1: {macro_f1:.4f}")
    print()

    print(
        classification_report(
            labels,
            predictions
        )
    )


if __name__ == "__main__":
    main()