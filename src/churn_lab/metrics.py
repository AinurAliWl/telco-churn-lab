"""Evaluation and campaign assumptions are kept separate from model fitting."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_indices(scores, ids, fraction: float) -> np.ndarray:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    score = np.asarray(scores, dtype=float)
    identity = np.asarray(ids, dtype=str)
    if len(score) != len(identity) or not np.isfinite(score).all():
        raise ValueError("Scores and IDs must align and all scores must be finite.")
    count = math.ceil(len(score) * fraction) if fraction else 0
    # Deterministic tie handling; labels are never used to break ties.
    return np.lexsort((identity, -score))[:count]


def ranking_metrics(y, scores, ids, fraction: float) -> dict:
    target = np.asarray(y, dtype=int)
    selected = top_indices(scores, ids, fraction)
    found = int(target[selected].sum())
    precision = found / len(selected) if len(selected) else 0.0
    prevalence = float(target.mean()) if len(target) else 0.0
    return {
        "fraction": fraction,
        "selected": len(selected),
        "churners_found": found,
        "precision_at_k": precision,
        "recall_at_k": found / int(target.sum()) if target.sum() else 0.0,
        "lift_at_k": precision / prevalence if prevalence else 0.0,
    }


def classification_metrics(y, probability, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = p >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def choose_threshold(y, p) -> float:
    """A secondary F1 operating point; choose on policy data only."""
    thresholds = np.linspace(0.05, 0.95, 91)
    quality = [f1_score(y, np.asarray(p) >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(quality))])


def bootstrap_intervals(y, p, ids, *, repetitions: int = 400, seed: int = 81) -> dict:
    """Fixed-model, stratified bootstrap; does not estimate retraining uncertainty."""
    y, p, ids = np.asarray(y), np.asarray(p), np.asarray(ids)
    groups = [np.flatnonzero(y == value) for value in [0, 1]]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Bootstrap evaluation needs both target classes.")
    rng = np.random.default_rng(seed)
    values = {
        key: [] for key in ["roc_auc", "average_precision", "recall_at_10pct", "lift_at_10pct"]
    }
    for _ in range(repetitions):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        ranks = ranking_metrics(y[ix], p[ix], ids[ix], 0.1)
        values["roc_auc"].append(roc_auc_score(y[ix], p[ix]))
        values["average_precision"].append(average_precision_score(y[ix], p[ix]))
        values["recall_at_10pct"].append(ranks["recall_at_k"])
        values["lift_at_10pct"].append(ranks["lift_at_k"])
    return {key: [float(x) for x in np.quantile(v, [0.025, 0.975])] for key, v in values.items()}


def paired_ap_interval(y, challenger, baseline, *, repetitions=400, seed=82) -> dict:
    y = np.asarray(y)
    a, b = np.asarray(challenger), np.asarray(baseline)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in [0, 1]]
    deltas = []
    for _ in range(repetitions):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(average_precision_score(y[ix], a[ix]) - average_precision_score(y[ix], b[ix]))
    return {
        "delta_ap": float(average_precision_score(y, a) - average_precision_score(y, b)),
        "ci95": [float(v) for v in np.quantile(deltas, [0.025, 0.975])],
    }


def campaign_plan(
    frame: pd.DataFrame,
    probability,
    *,
    fraction=0.1,
    rescue_rate=0.15,
    horizon=6,
    margin=0.6,
    contact_cost=1.0,
    incentive_cost=5.0,
    strategy="risk",
    positive_only=False,
) -> tuple[pd.DataFrame, dict]:
    """Illustrative constant rescue rate conditional on counterfactual churn; not uplift."""
    if not 0 <= rescue_rate <= 1 or not 0 <= margin <= 1:
        raise ValueError("rescue_rate and margin must be in [0, 1].")
    if (
        not np.isfinite([horizon, contact_cost, incentive_cost]).all()
        or min(horizon, contact_cost, incentive_cost) < 0
    ):
        raise ValueError("Horizon and costs must be finite and non-negative.")
    if strategy not in {"risk", "expected_value", "monthly_charge"}:
        raise ValueError("Unknown campaign strategy.")
    p = np.asarray(probability, dtype=float)
    if len(p) != len(frame) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must align with input rows and be in [0, 1].")
    result = frame[["customerID", "MonthlyCharges"]].copy().reset_index(drop=True)
    result["churn_probability"] = p
    result["assumed_saved_margin"] = p * rescue_rate * result.MonthlyCharges * horizon * margin
    result["campaign_cost"] = contact_cost + incentive_cost
    result["expected_net_value"] = result.assumed_saved_margin - result.campaign_cost
    score = {
        "risk": p,
        "expected_value": result.expected_net_value,
        "monthly_charge": result.MonthlyCharges,
    }[strategy]
    ix = top_indices(score, result.customerID, fraction)
    chosen = result.iloc[ix].copy()
    if positive_only:
        chosen = chosen[chosen.expected_net_value > 0].copy()
    chosen.insert(0, "priority", np.arange(1, len(chosen) + 1))
    total_cost = float(chosen.campaign_cost.sum())
    summary = {
        "contacted": len(chosen),
        "expected_churners": float(chosen.churn_probability.sum()),
        "assumed_saved_customers": float(chosen.churn_probability.sum() * rescue_rate),
        "campaign_cost": total_cost,
        "expected_net_value": float(chosen.expected_net_value.sum()),
        "assumed_saved_margin": float(chosen.assumed_saved_margin.sum()),
    }
    return chosen, summary
