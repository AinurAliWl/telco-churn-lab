"""The delivered fitted model must reproduce the published holdout predictions."""

import json

import numpy as np
import pandas as pd
import pytest

from churn_lab.data import FEATURES, ROOT
from churn_lab.modeling import load_bundle, prepare_features, probabilities


@pytest.fixture(scope="module")
def saved():
    if not (ROOT / "artifacts/manifest.json").exists():
        pytest.skip("Run churn-lab train to produce integration artifacts")
    return load_bundle(), pd.read_parquet(ROOT / "reports/holdout_customers.parquet")


def test_model_reproduces_saved_predictions(saved):
    bundle, frame = saved
    saved_p = pd.read_parquet(ROOT / "reports/holdout_predictions.parquet")
    assert frame.customerID.tolist() == saved_p.customerID.tolist()
    for name in bundle["estimators"]:
        np.testing.assert_allclose(probabilities(bundle, frame, name), saved_p[name], atol=1e-10)


def test_medians_only_use_training_rows(saved):
    bundle, _ = saved
    all_data = pd.read_parquet(ROOT / "reports/features.parquet")
    training = all_data.query("split == 'train'")
    for feature, median in bundle["medians"].items():
        assert median == pytest.approx(training[feature].median())


def test_champion_selected_from_policy_not_test(saved):
    bundle, _ = saved
    selection = pd.read_csv(ROOT / "reports/model_selection_policy.csv")
    expected = selection.sort_values("average_precision", ascending=False).iloc[0].model
    assert bundle["champion"] == expected


def test_shap_additivity_matches_raw_catboost_score(saved):
    bundle, frame = saved
    shap = np.load(ROOT / "reports/shap_values.npz")["values"]
    x = prepare_features(frame, bundle["medians"])[FEATURES]
    raw_prediction = bundle["estimators"]["catboost_sql"].predict(
        x, prediction_type="RawFormulaVal"
    )
    np.testing.assert_allclose(shap.sum(axis=1), raw_prediction, atol=1e-8)


def test_source_dataset_matches_published_manifest(saved):
    from churn_lab.data import DATA_SHA256

    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    assert manifest["source_sha256"] == DATA_SHA256
