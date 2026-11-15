"""Data contracts and target independence protect the experiment boundary."""

import pandas as pd
import pytest

from churn_lab.data import FEATURES, ROOT, feature_frame, make_splits, validate_raw


@pytest.fixture
def raw():
    return pd.read_csv(ROOT / "data/raw/telco.csv", dtype=str, keep_default_na=False)


def test_dataset_contract_and_unique_ids(raw):
    clean = validate_raw(raw)
    assert len(clean) == 7043
    assert clean.customerID.nunique() == len(clean)


def test_sql_features_do_not_depend_on_labels(raw):
    original = feature_frame(raw.iloc[:50])
    flipped = raw.iloc[:50].copy()
    flipped["Churn"] = flipped.Churn.map({"Yes": "No", "No": "Yes"})
    pd.testing.assert_frame_equal(original[FEATURES], feature_frame(flipped)[FEATURES])


def test_training_and_inference_sql_are_identical(raw):
    training = feature_frame(raw.iloc[:50])
    inference = feature_frame(raw.iloc[:50].drop(columns="Churn"), training=False)
    pd.testing.assert_frame_equal(training.drop(columns="target"), inference)


def test_extra_target_is_ignored_at_inference(raw):
    a = raw.iloc[:10].copy()
    b = a.copy()
    b["Churn"] = "arbitrary label"
    pd.testing.assert_frame_equal(
        feature_frame(a, training=False), feature_frame(b, training=False)
    )


def test_splits_are_disjoint_complete_and_reproducible(raw):
    data = feature_frame(raw)
    a = make_splits(data)
    pd.testing.assert_frame_equal(a, make_splits(data))
    assert a.customerID.is_unique
    assert set(a.customerID) == set(data.customerID)
    assert set(a.split) == {"train", "calibration", "policy", "test"}
    for name, fraction in [("train", 0.6), ("calibration", 0.1), ("policy", 0.1), ("test", 0.2)]:
        group = data.loc[a.split == name]
        assert abs(len(group) / len(data) - fraction) < 0.001
        assert abs(group.target.mean() - data.target.mean()) < 0.003


def test_blank_total_charges_remain_missing_until_train_imputation(raw):
    out = feature_frame(raw)
    assert out.TotalCharges.isna().sum() == 11
    assert out.total_charges_missing.sum() == 11


def test_demographics_target_and_id_are_not_model_features():
    assert not set(FEATURES) & {
        "target",
        "Churn",
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "split",
    }


@pytest.mark.parametrize(
    "column,value",
    [
        ("MonthlyCharges", "-4"),
        ("MonthlyCharges", "inf"),
        ("tenure", "1.5"),
        ("tenure", ""),
        ("TotalCharges", "bad"),
        ("Contract", "unknown"),
        ("Churn", "1"),
    ],
)
def test_invalid_values_fail_with_actionable_error(raw, column, value):
    small = raw.iloc[:4].copy()
    small.loc[small.index[0], column] = value
    with pytest.raises(ValueError, match=column):
        validate_raw(small)


def test_duplicate_ids_rejected(raw):
    with pytest.raises(ValueError, match="unique"):
        validate_raw(pd.concat([raw.iloc[:2], raw.iloc[:2]]))


def test_missing_columns_rejected(raw):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw(raw.drop(columns="Contract"))


def test_empty_batch_rejected(raw):
    with pytest.raises(ValueError, match="no customer"):
        validate_raw(raw.iloc[:0])
