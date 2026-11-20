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


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, normalise headline text, and retain all labels for auditing.

    Duplicate labels must be assessed across *all* splits, not independently:
    removing/retaining a key based only on one split can hide contradictions.
    """
    if not {"sentence", "label"}.issubset(frame.columns):
        raise ValueError("Dataset must have 'sentence' and 'label' columns")
    df = frame.loc[:, ["sentence", "label"]].copy()
    if df["sentence"].isna().any() or df["label"].isna().any():
        raise ValueError("Missing sentences or labels: fix the data before training")
    if not df["sentence"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Every sentence must be a string")
    if not df["label"].map(
        lambda value: type(value) is int
        or (hasattr(value, "item") and type(value.item()) is int)
    ).all():
        raise ValueError("Labels must be integer class IDs 0, 1 or 2")
    df["label"] = df["label"].astype(int)
    if not df["label"].isin(LABEL_NAMES).all():
        raise ValueError("Labels must be integer class IDs 0, 1 or 2")
    df["sentence"] = df["sentence"].map(lambda s: re.sub(r"\s+", " ", s).strip())
    df["key"] = df["sentence"].map(_key)
    if (df["key"] == "").any():
        raise ValueError("Empty headline after whitespace normalisation")
    return df.reset_index(drop=True)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean a standalone frame, rejecting ambiguous labels (strict helper).

    Use prepare_splits for real datasets: it drops contradictory headlines
    from all splits with an explicit audit trail instead of failing training.
    """
    df = _validate_frame(frame)
    conflicts = df.groupby("key")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting sentiment labels for identical headlines")
    return df.drop_duplicates("key").reset_index(drop=True)


def prepare_splits(raw: Mapping[str, pd.DataFrame], seed: int = SEED) -> tuple[dict[str, pd.DataFrame], dict]:
    """Prepare leak-free splits and quarantine ambiguous source annotations.

    Exact same headline with different labels is excluded from *every* split;
    neither majority voting nor arbitrary label selection is justified. All
    exclusions and duplicates are counted in the returned audit metadata.
    When supplied, upstream test > validation > train priority is preserved for
    exact-text overlap. Missing splits trigger a seeded 70/15/15 split.
    """
    if "train" not in raw:
        raise ValueError("Dataset has no 'train' split")
    source = "provided" if {"train", "validation", "test"}.issubset(raw) else "generated_70_15_15"
    names = ("train", "validation", "test") if source == "provided" else tuple(sorted(raw))
    validated = {name: _validate_frame(raw[name]) for name in names}
    original_counts = {name: len(frame) for name, frame in validated.items()}

    # Compare annotations globally: contradictions can be within or across splits.
    all_labels = pd.concat([frame[["key", "label"]] for frame in validated.values()], ignore_index=True)
    ambiguous_keys = set(all_labels.groupby("key")["label"].nunique().loc[lambda counts: counts > 1].index)
    ambiguous_rows = {
        name: int(frame["key"].isin(ambiguous_keys).sum())
        for name, frame in validated.items()
    }
    filtered = {
        name: frame.loc[~frame["key"].isin(ambiguous_keys)].drop_duplicates("key").reset_index(drop=True)
        for name, frame in validated.items()
    }

    if source == "provided":
        splits = filtered
    else:
        combined = pd.concat(list(filtered.values()), ignore_index=True).drop_duplicates("key")
        train_val, test = train_test_split(combined, test_size=0.15, random_state=seed, stratify=combined["label"])
        train, validation = train_test_split(train_val, test_size=0.15 / 0.85, random_state=seed, stratify=train_val["label"])
        splits = {"train": train, "validation": validation, "test": test}

    pre_counts = {name: len(frame) for name, frame in splits.items()}
    test_keys = set(splits["test"]["key"])
    splits["validation"] = splits["validation"].loc[~splits["validation"]["key"].isin(test_keys)]
    holdout_keys = test_keys | set(splits["validation"]["key"])
    splits["train"] = splits["train"].loc[~splits["train"]["key"].isin(holdout_keys)]
    for name, frame in splits.items():
        if len(frame) == 0 or set(frame["label"]) != set(LABEL_NAMES):
            raise ValueError(f"'{name}' split is empty or missing a sentiment class after cleaning")
        splits[name] = frame.loc[:, ["sentence", "label"]].reset_index(drop=True)
    meta = {
        "source": source,
        "seed": seed,
        "ambiguity_policy": "exclude_all_conflicting_headlines",
        "ambiguous_unique_headlines_excluded": len(ambiguous_keys),
        "ambiguous_source_rows_excluded_by_split": ambiguous_rows,
        "original_source_rows_by_split": original_counts,
        "rows_before_cross_split_dedup": pre_counts,
        "rows_after_cross_split_dedup": {name: len(frame) for name, frame in splits.items()},
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
