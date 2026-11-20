import pandas as pd
import pytest

from finsight.data.splits import clean_frame, prepare_splits


def frame(prefix, n=12):
    return pd.DataFrame({"sentence": [f"{prefix} unique financial headline {i}" for i in range(n)], "label": [i % 3 for i in range(n)]})


def test_clean_frame_drops_whitespace_duplicates():
    df = pd.DataFrame({"sentence": [" Strong  earnings ", "strong earnings", "Lower revenue"], "label": [2, 2, 0]})
    cleaned = clean_frame(df)
    assert len(cleaned) == 2
    assert cleaned.iloc[0]["sentence"] == "Strong earnings"


def test_reject_conflicting_duplicate_labels():
    with pytest.raises(ValueError, match="Conflicting"):
        clean_frame(pd.DataFrame({"sentence": ["Same", "same"], "label": [0, 1]}))


def test_reject_bad_labels():
    with pytest.raises(ValueError, match="Labels"):
        clean_frame(pd.DataFrame({"sentence": ["Example"], "label": [4]}))


def test_provided_splits_are_decontaminated():
    train, validation, test = frame("train"), frame("validation"), frame("test")
    train.loc[0, "sentence"] = validation.loc[0, "sentence"]
    train.loc[0, "label"] = validation.loc[0, "label"]
    result, meta = prepare_splits({"train": train, "validation": validation, "test": test})
    assert len(result["train"]) == 11
    assert meta["source"] == "provided"
    sets = [set(df["sentence"].str.casefold()) for df in result.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_fallback_is_deterministic_and_stratified():
    source = frame("generated", 90)
    a, am = prepare_splits({"train": source})
    b, bm = prepare_splits({"train": source})
    assert am["source"] == "generated_70_15_15"
    assert am == bm
    for name in ("train", "validation", "test"):
        assert a[name].equals(b[name])
        assert set(a[name]["label"]) == {0, 1, 2}
