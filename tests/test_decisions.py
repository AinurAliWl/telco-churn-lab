import numpy as np
import pandas as pd
import pytest

from churn_lab.metrics import campaign_plan, ranking_metrics, top_indices


def test_rankings_use_scores_and_id_ties_without_labels():
    scores, ids = [0.8, 0.8, 0.1], ["B", "A", "C"]
    assert top_indices(scores, ids, 0.34).tolist() == [1, 0]


def test_budget_coverage_known_example():
    result = ranking_metrics([1, 1, 0, 0, 0], [0.9, 0.8, 0.4, 0.2, 0.1], list("ABCDE"), 0.2)
    assert result["selected"] == 1
    assert result["precision_at_k"] == 1
    assert result["recall_at_k"] == 0.5
    assert result["lift_at_k"] == 2.5


def test_zero_capacity_contacts_nobody():
    assert len(top_indices([0.8], ["A"], 0)) == 0


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_invalid_capacity_rejected(fraction):
    with pytest.raises(ValueError, match="fraction"):
        top_indices([0.8], ["A"], fraction)


def customers():
    return pd.DataFrame(
        {"customerID": ["A", "B"], "MonthlyCharges": [50.0, 100.0], "target": [1, 0]}
    )


def test_scenario_arithmetic_and_unconditional_incentive_cost():
    plan, summary = campaign_plan(
        customers(),
        [0.8, 0.1],
        fraction=0.5,
        rescue_rate=0.2,
        horizon=6,
        margin=0.5,
        contact_cost=1,
        incentive_cost=5,
    )
    assert plan.customerID.tolist() == ["A"]
    assert summary["campaign_cost"] == 6
    assert summary["assumed_saved_customers"] == pytest.approx(0.16)
    assert summary["expected_net_value"] == pytest.approx(18)


def test_scenario_does_not_read_observed_churn_labels():
    a, sa = campaign_plan(customers(), [0.8, 0.1], fraction=1)
    flipped = customers().assign(target=[0, 1])
    b, sb = campaign_plan(flipped, [0.8, 0.1], fraction=1)
    pd.testing.assert_frame_equal(a, b)
    assert sa == sb


def test_no_assumed_rescue_means_cost_only_or_no_contacts():
    _, loss = campaign_plan(customers(), [0.8, 0.1], fraction=1, rescue_rate=0, positive_only=False)
    assert loss["expected_net_value"] == -12
    plan, empty = campaign_plan(
        customers(), [0.8, 0.1], fraction=1, rescue_rate=0, positive_only=True
    )
    assert plan.empty
    assert empty["expected_net_value"] == 0


def test_value_strategy_can_choose_a_different_customer():
    frame = customers().assign(MonthlyCharges=[10.0, 100.0])
    risk, _ = campaign_plan(frame, [0.8, 0.5], fraction=0.5, strategy="risk")
    value, _ = campaign_plan(frame, [0.8, 0.5], fraction=0.5, strategy="expected_value")
    assert risk.customerID.iloc[0] == "A"
    assert value.customerID.iloc[0] == "B"


@pytest.mark.parametrize("probabilities", [[-0.1, 0.2], [float("nan"), 0.2], [0.2]])
def test_invalid_predictions_rejected(probabilities):
    with pytest.raises(ValueError, match="Probabilities"):
        campaign_plan(customers(), probabilities)


@pytest.mark.parametrize(
    "params",
    [{"rescue_rate": 1.1}, {"margin": -0.1}, {"contact_cost": -1}, {"horizon": float("nan")}],
)
def test_invalid_business_assumptions_rejected(params):
    with pytest.raises(ValueError):
        campaign_plan(customers(), [0.8, 0.1], **params)


def test_recall_of_full_contact_list_is_one():
    result = ranking_metrics([0, 1, 0, 1], np.linspace(0, 1, 4), list("ABCD"), 1)
    assert result["recall_at_k"] == 1
