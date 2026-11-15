"""Generate figures and a results note from actual saved experiment outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from churn_lab.data import ROOT
from churn_lab.metrics import top_indices

COLORS = ["#137c6b", "#bd6539", "#4c6b9a"]
LABELS = {
    "catboost_sql": "CatBoost + SQL features",
    "catboost_base": "CatBoost · base features",
    "logistic_regression": "Logistic regression",
}


def write_reports(root: Path = ROOT):
    report = root / "reports"
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "artifacts/manifest.json").read_text())
    quality = json.loads((report / "data_quality.json").read_text())
    uncertainty = json.loads((report / "uncertainty.json").read_text())
    pred = pd.read_parquet(report / "holdout_predictions.parquet")
    scores = pd.read_csv(report / "test_metrics.csv")
    budgets = pd.read_csv(report / "budget_metrics.csv")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f5ef",
            "axes.facecolor": "#f6f5ef",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.labelcolor": "#172e2c",
            "text.color": "#172e2c",
            "savefig.dpi": 160,
        }
    )
    names = ["logistic_regression", "catboost_base", "catboost_sql"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    for name, color in zip(names, COLORS, strict=True):
        precision, recall, _ = precision_recall_curve(pred.target, pred[name])
        ap = scores.loc[scores.model == name, "average_precision"].iloc[0]
        ax.plot(recall, precision, label=f"{LABELS[name]} · AP {ap:.3f}", color=color)
    ax.axhline(
        pred.target.mean(), color="#9a9a8b", linestyle="--", label="Random-selection reference"
    )
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Identify churn within a limited contact capacity",
    )
    ax.legend(fontsize=8)
    fig.savefig(figures / "precision_recall.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    for name, color in zip(names, COLORS, strict=True):
        order = top_indices(pred[name], pred.customerID, 1)
        gain = np.r_[0, np.cumsum(pred.target.to_numpy()[order]) / pred.target.sum()]
        ax.plot(np.linspace(0, 1, len(gain)), gain, label=LABELS[name], color=color)
    ax.plot([0, 1], [0, 1], "--", color="#9a9a8b", label="Random-selection reference")
    ax.axvline(0.1, color="#172e2c", alpha=0.35)
    ax.set(
        xlabel="Share of customers contacted",
        ylabel="Share of churners captured",
        title="Cumulative gains · fixed holdout",
    )
    ax.legend(fontsize=8)
    fig.savefig(figures / "cumulative_gains.png")
    plt.close(fig)

    champion = meta["champion"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    for suffix, label, color in [
        ("_raw", "Before calibration", COLORS[1]),
        ("", "Selected probability output", COLORS[0]),
    ]:
        observed, estimated = calibration_curve(
            pred.target, pred[champion + suffix], n_bins=8, strategy="quantile"
        )
        ax.plot(estimated, observed, marker="o", color=color, label=label)
    ax.plot([0, 1], [0, 1], "--", color="#9a9a8b")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed churn fraction",
        title=f"Reliability · {LABELS[champion]}",
    )
    ax.legend(fontsize=8)
    fig.savefig(figures / "calibration.png")
    plt.close(fig)

    importance = pd.read_csv(report / "shap_importance.csv").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    ax.barh(importance.feature, importance.mean_abs_shap, color=COLORS[0])
    ax.set(
        xlabel="Mean absolute SHAP value · raw log-odds",
        title="How CatBoost + SQL features forms its predictions",
    )
    fig.savefig(figures / "shap_importance.png")
    plt.close(fig)

    top = budgets[(budgets.model == champion) & np.isclose(budgets.fraction, 0.1)].iloc[0]
    interval = uncertainty["per_model"][champion]
    delta = uncertainty["catboost_sql_minus_logistic_ap"]
    md = f"""# Experiment results

Author: Ainur Ali

## Frozen evaluation protocol

- Source: IBM Telco sample, {quality["rows"]:,} customers; verified SHA-256.
- Split: 60% training / 10% calibration / 10% policy / 20% test, stratified by label.
- Three-fold CV and Optuna search run inside training only ({meta["optuna_trials"]} trials).
- Calibration is fitted on its own split. The policy split selects calibration, an F1 threshold and the model with the highest average precision.
- The champion was frozen before reading test labels: **{champion}**.
- Demographics are retained for descriptive auditing and excluded from model features. This does not establish fairness or remove correlated proxies.

## Holdout results

| Model | ROC-AUC | Average precision | Brier score (lower is better) |
| --- | ---: | ---: | ---: |
"""
    for row in scores.itertuples():
        md += (
            f"| {row.model} | {row.roc_auc:.4f} | {row.average_precision:.4f} | {row.brier:.4f} |\n"
        )
    md += f"""
For the selected model, contacting the top 10% means selecting **{int(top.selected)} of {len(pred):,} held-out customers**. This captures **{int(top.churners_found)} of {int(pred.target.sum())} churners**: precision **{top.precision_at_k:.1%}**, recall **{top.recall_at_k:.1%}**, lift **{top.lift_at_k:.2f}×** over the holdout prevalence.

The 95% fixed-model bootstrap interval is {interval["roc_auc"][0]:.3f}–{interval["roc_auc"][1]:.3f} for ROC-AUC and {interval["average_precision"][0]:.3f}–{interval["average_precision"][1]:.3f} for average precision ({meta["bootstrap_repetitions"]} stratified resamples).

CatBoost + SQL minus Logistic Regression: AP difference **{delta["delta_ap"]:+.4f}**, paired 95% interval **[{delta["ci95"][0]:+.4f}, {delta["ci95"][1]:+.4f}]**. An interval crossing zero does not support a clear improvement on this holdout.

## What the result supports

The experiment measures ranking, classification and probability reliability on an illustrative customer snapshot. It does not measure customers saved, causal offer effects, future-month performance or performance on Beeline data. Missing observation timestamps prevent a realistic out-of-time validation.

The base-feature CatBoost comparison uses the same hyperparameters selected for the SQL-feature model: it is a fixed-configuration ablation, not a separately optimized benchmark. SHAP explains the CatBoost + SQL model in raw log-odds; it does not identify causes of churn or effects of changing a contract.

The campaign planner uses user-specified assumptions for rescue probability, contribution margin, horizon and costs. The rescue probability is conditional on a customer otherwise churning; no such treatment outcome is observed here. The incentive cost is assumed paid for every contacted customer. Predicted customer value is an illustrative planning quantity, not measured ROI or customer lifetime value.

Bootstrap intervals condition on the fixed fitted models and observed class balance. They exclude retraining uncertainty and future distribution shift. Segment results are descriptive, with small groups suppressed; they are not evidence of causal effects.

## Figures

![Precision and recall](figures/precision_recall.png)

![Cumulative gains](figures/cumulative_gains.png)

![Probability calibration](figures/calibration.png)

![CatBoost explanation](figures/shap_importance.png)
"""
    (report / "RESULTS.md").write_text(md, encoding="utf-8")
