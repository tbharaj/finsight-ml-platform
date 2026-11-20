"""Validate financial-headline data and prevent exact-text split leakage."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_NAME = "lmassaron/FinancialPhraseBank"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
SEED = 42


def _key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip().casefold()


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return unique, valid (sentence, label) examples; reject ambiguous labels."""
    if not {"sentence", "label"}.issubset(frame.columns):
        raise ValueError("Dataset must have 'sentence' and 'label' columns")
    df = frame.loc[:, ["sentence", "label"]].copy()
    if df["sentence"].isna().any() or df["label"].isna().any():
        raise ValueError("Missing sentences or labels: fix the data before training")
    if not df["sentence"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Every sentence must be a string")
    if not df["label"].map(lambda value: type(value) is int or isinstance(value, (int,)) or (hasattr(value, "item") and isinstance(value.item(), int))).all():
        raise ValueError("Labels must be integer class IDs 0, 1 or 2")
    df["label"] = df["label"].astype(int)
    if not df["label"].isin(LABEL_NAMES).all():
        raise ValueError("Labels must be integer class IDs 0, 1 or 2")
    df["sentence"] = df["sentence"].map(lambda s: re.sub(r"\s+", " ", s).strip())
    df["key"] = df["sentence"].map(_key)
    if (df["key"] == "").any():
        raise ValueError("Empty headline after whitespace normalisation")
    conflicts = df.groupby("key")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting sentiment labels for identical headlines")
    return df.drop_duplicates("key").reset_index(drop=True)


def prepare_splits(raw: Mapping[str, pd.DataFrame], seed: int = SEED) -> tuple[dict[str, pd.DataFrame], dict]:
    """Use official splits when complete; otherwise create seeded 70/15/15 splits.

    Test and validation text take priority during decontamination. Source split
    metadata and dropped-row counts are returned for auditability.
    """
    if "train" not in raw:
        raise ValueError("Dataset has no 'train' split")
    source = "provided" if {"train", "validation", "test"}.issubset(raw) else "generated_70_15_15"
    if source == "provided":
        splits = {name: clean_frame(raw[name]) for name in ("train", "validation", "test")}
    else:
        combined = pd.concat([clean_frame(frame) for frame in raw.values()], ignore_index=True)
        combined = clean_frame(combined)
        train_val, test = train_test_split(combined, test_size=0.15, random_state=seed, stratify=combined["label"])
        train, validation = train_test_split(train_val, test_size=0.15 / 0.85, random_state=seed, stratify=train_val["label"])
        splits = {"train": train, "validation": validation, "test": test}

    # Identical headlines must never occur across the train/validation/test boundary.
    label_sets = pd.concat([df.loc[:, ["key", "label"]] for df in splits.values()])
    if (label_sets.groupby("key")["label"].nunique() > 1).any():
        raise ValueError("Conflicting labels across splits; resolve them at the source")
    pre_counts = {name: len(df) for name, df in splits.items()}
    test_keys = set(splits["test"]["key"])
    splits["validation"] = splits["validation"].loc[~splits["validation"]["key"].isin(test_keys)]
    holdout_keys = test_keys | set(splits["validation"]["key"])
    splits["train"] = splits["train"].loc[~splits["train"]["key"].isin(holdout_keys)]
    for name, frame in splits.items():
        if len(frame) == 0 or set(frame["label"]) != set(LABEL_NAMES):
            raise ValueError(f"'{name}' split is empty or missing a sentiment class")
        splits[name] = frame.loc[:, ["sentence", "label"]].reset_index(drop=True)
    meta = {
        "source": source,
        "seed": seed,
        "rows_before_cross_split_dedup": pre_counts,
        "rows_after_cross_split_dedup": {name: len(df) for name, df in splits.items()},
    }
    return splits, meta


def load_splits(dataset_name: str = DATASET_NAME) -> tuple[dict[str, pd.DataFrame], dict]:
    """Download the source dataset (requires the optional `data` dependency)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
    ds = load_dataset(dataset_name)
    frames = {name: ds[name].to_pandas() for name in ds.keys()}
    splits, meta = prepare_splits(frames)
    meta["dataset"] = dataset_name
    return splits, meta
