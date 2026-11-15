"""Model search, independent calibration and a frozen holdout evaluation."""

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_lab.data import CATEGORICAL, FEATURES, NUMERIC, ROOT, build_data, sha256
from churn_lab.metrics import (
    bootstrap_intervals,
    choose_threshold,
    classification_metrics,
    paired_ap_interval,
    ranking_metrics,
)

ENGINEERED = ["addon_count", "total_charges_missing", "tenure_band", "contract_internet"]
MODEL_NAMES = ["logistic_regression", "catboost_base", "catboost_sql"]


def prepare_features(frame: pd.DataFrame, medians: dict) -> pd.DataFrame:
    result = frame[FEATURES].copy()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors="raise").fillna(medians[col]).astype(float)
    for col in CATEGORICAL:
        result[col] = result[col].astype(str)
    return result


def logistic_pipeline(c: float) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("prepare", preprocess),
            ("model", LogisticRegression(C=c, max_iter=2500, solver="lbfgs")),
        ]
    )


def logit(p) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def probabilities(bundle: dict, frame: pd.DataFrame, name: str, *, mode="selected") -> np.ndarray:
    if name not in bundle["estimators"]:
        raise ValueError(f"Unknown model: {name}")
    x = prepare_features(frame, bundle["medians"])
    raw = bundle["estimators"][name].predict_proba(x[bundle["feature_map"][name]])[:, 1]
    use = mode == "calibrated" or (
        mode == "selected" and bundle["metadata"][name]["use_calibration"]
    )
    return bundle["calibrators"][name].predict_proba(logit(raw))[:, 1] if use else raw


def load_bundle(root: Path = ROOT) -> dict:
    """Load the bundled local artifact only; never deserialize an uploaded model."""
    artifact = root / "artifacts/bundle.joblib"
    manifest = json.loads((root / "artifacts/manifest.json").read_text())
    if sha256(artifact) != manifest["bundle_sha256"]:
        raise ValueError("Model artifact checksum mismatch. Restore or retrain the model.")
    return joblib.load(artifact)


def train_project(*, out: Path, trials=16, bootstrap=400, seed=42, threads=2) -> dict:
    if trials < 1 or bootstrap < 20 or threads < 1:
        raise ValueError("Use at least 1 trial, 20 bootstrap repetitions and 1 thread.")
    started = time.perf_counter()
    artifact_dir = out / "artifacts"
    report_dir = out / "reports"
    for folder in [artifact_dir, report_dir, report_dir / "figures"]:
        folder.mkdir(parents=True, exist_ok=True)
    frame = build_data(report_dir, seed)
    partitions = {
        name: frame[frame.split == name].reset_index(drop=True)
        for name in ["train", "calibration", "policy", "test"]
    }
    train = partitions["train"]
    # Logistic preprocessing is fitted inside each CV fold. CatBoost can use native numeric NaNs.
    x_raw, y = train[FEATURES].copy(), train.target
    for col in NUMERIC:
        x_raw[col] = x_raw[col].astype(float)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    lr_search = []
    for c in [0.01, 0.1, 1.0, 10.0]:
        scores = cross_val_score(
            logistic_pipeline(c), x_raw, y, cv=cv, scoring="average_precision", n_jobs=1
        )
        lr_search.append(
            {"C": c, "mean_cv_ap": float(scores.mean()), "std_cv_ap": float(scores.std())}
        )
    best_c = max(lr_search, key=lambda item: item["mean_cv_ap"])["C"]
    pd.DataFrame(lr_search).to_csv(report_dir / "logistic_search.csv", index=False)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=6)
    )

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 15.0, log=True),
            "iterations": trial.suggest_int("iterations", 250, 550, step=100),
        }
        scores = []
        for fit_ix, valid_ix in cv.split(x_raw, y):
            estimator = CatBoostClassifier(
                **params,
                random_seed=seed,
                verbose=False,
                thread_count=threads,
                allow_writing_files=False,
            )
            estimator.fit(x_raw.iloc[fit_ix], y.iloc[fit_ix], cat_features=CATEGORICAL)
            scores.append(
                average_precision_score(
                    y.iloc[valid_ix], estimator.predict_proba(x_raw.iloc[valid_ix])[:, 1]
                )
            )
        trial.set_user_attr("fold_ap", [float(v) for v in scores])
        print(f"Optuna trial {trial.number + 1}/{trials}: CV AP={np.mean(scores):.4f}", flush=True)
        return float(np.mean(scores))

    study.optimize(objective, n_trials=trials, n_jobs=1)
    study.trials_dataframe().to_csv(report_dir / "optuna_trials.csv", index=False)
    medians = {col: float(train[col].median()) for col in NUMERIC}
    bundle = {
        "estimators": {},
        "calibrators": {},
        "feature_map": {},
        "metadata": {},
        "medians": medians,
    }
    # Fit LogisticRegression's imputation/scaling from training rows only.
    bundle["estimators"]["logistic_regression"] = logistic_pipeline(best_c).fit(x_raw, y)
    bundle["feature_map"]["logistic_regression"] = FEATURES
    for name in ["catboost_base", "catboost_sql"]:
        feats = FEATURES if name == "catboost_sql" else [f for f in FEATURES if f not in ENGINEERED]
        estimator = CatBoostClassifier(
            **study.best_params,
            random_seed=seed,
            verbose=False,
            thread_count=threads,
            allow_writing_files=False,
        )
        # Final models use the same train-only median transformation as inference.
        estimator.fit(
            prepare_features(train, medians)[feats],
            y,
            cat_features=[f for f in CATEGORICAL if f in feats],
        )
        estimator.save_model(str(artifact_dir / f"{name}.cbm"))
        bundle["estimators"][name] = estimator
        bundle["feature_map"][name] = feats
    policy_rows = []
    for name in MODEL_NAMES:
        estimator, feats = bundle["estimators"][name], bundle["feature_map"][name]
        calibration = partitions["calibration"]
        raw_cal = estimator.predict_proba(prepare_features(calibration, medians)[feats])[:, 1]
        calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(
            logit(raw_cal), calibration.target
        )
        bundle["calibrators"][name] = calibrator
        policy = partitions["policy"]
        raw_policy = estimator.predict_proba(prepare_features(policy, medians)[feats])[:, 1]
        cal_policy = calibrator.predict_proba(logit(raw_policy))[:, 1]
        raw_brier = brier_score_loss(policy.target, raw_policy)
        cal_brier = brier_score_loss(policy.target, cal_policy)
        # Independent policy split selects calibration and threshold, before test access.
        use = bool(cal_brier < raw_brier and calibrator.coef_[0, 0] > 0)
        chosen = cal_policy if use else raw_policy
        threshold = choose_threshold(policy.target, chosen)
        bundle["metadata"][name] = {"use_calibration": use, "threshold": threshold}
        policy_rows.append(
            {
                "model": name,
                "average_precision": float(average_precision_score(policy.target, chosen)),
                "raw_brier": raw_brier,
                "calibrated_brier": cal_brier,
                **bundle["metadata"][name],
            }
        )
    policy_table = pd.DataFrame(policy_rows)
    champion = str(policy_table.sort_values("average_precision", ascending=False).iloc[0].model)
    bundle["champion"] = champion
    policy_table.to_csv(report_dir / "model_selection_policy.csv", index=False)
    # Selection is now frozen. All following label-based reporting uses the untouched test.
    test = partitions["test"]
    test.to_parquet(report_dir / "holdout_customers.parquet", index=False)
    test_predictions = test[
        [
            "customerID",
            "target",
            "MonthlyCharges",
            "Contract",
            "InternetService",
            "tenure_band",
            "gender",
            "SeniorCitizen",
        ]
    ].copy()
    metric_rows, budget_rows, intervals = [], [], {}
    prevalence = float(train.target.mean())
    for name in MODEL_NAMES + ["constant_baseline"]:
        raw_p = (
            probabilities(bundle, test, name, mode="raw")
            if name != "constant_baseline"
            else np.full(len(test), prevalence)
        )
        p = probabilities(bundle, test, name) if name != "constant_baseline" else raw_p
        threshold = bundle["metadata"][name]["threshold"] if name != "constant_baseline" else 0.5
        metric_rows.append(
            {
                "model": name,
                **classification_metrics(test.target, p, threshold),
                "raw_brier": float(brier_score_loss(test.target, raw_p)),
            }
        )
        test_predictions[name] = p
        if name != "constant_baseline":
            test_predictions[f"{name}_raw"] = raw_p
            test_predictions[f"{name}_calibrated"] = probabilities(
                bundle, test, name, mode="calibrated"
            )
        for fraction in [0.05, 0.10, 0.20]:
            budget_rows.append(
                {"model": name, **ranking_metrics(test.target, p, test.customerID, fraction)}
            )
        if name != "constant_baseline":
            intervals[name] = bootstrap_intervals(
                test.target, p, test.customerID, repetitions=bootstrap, seed=seed + 20
            )
    for fraction in [0.05, 0.10, 0.20]:
        budget_rows.append(
            {
                "model": "monthly_charge_heuristic",
                **ranking_metrics(test.target, test.MonthlyCharges, test.customerID, fraction),
            }
        )
    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(report_dir / "test_metrics.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(report_dir / "budget_metrics.csv", index=False)
    test_predictions.to_parquet(report_dir / "holdout_predictions.parquet", index=False)
    paired = paired_ap_interval(
        test.target,
        test_predictions.catboost_sql,
        test_predictions.logistic_regression,
        repetitions=bootstrap,
        seed=seed + 21,
    )
    uncertainty = {
        "method": "fixed-model stratified bootstrap; 95% percentile intervals",
        "repetitions": bootstrap,
        "per_model": intervals,
        "catboost_sql_minus_logistic_ap": paired,
    }
    (report_dir / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2) + "\n")
    # Audit samples are descriptive; do not tune models from these holdout groups.
    groups = []
    for col in ["Contract", "InternetService", "tenure_band", "gender", "SeniorCitizen"]:
        for group, subset in test.groupby(col):
            if len(subset) < 40:
                continue
            positions = test.index.get_indexer(subset.index)
            p = test_predictions[champion].to_numpy()[positions]
            groups.append(
                {
                    "dimension": col,
                    "group": str(group),
                    "n": len(subset),
                    "churn_rate": subset.target.mean(),
                    "mean_prediction": p.mean(),
                    "brier": brier_score_loss(subset.target, p),
                    "recall_at_frozen_threshold": classification_metrics(
                        subset.target, p, bundle["metadata"][champion]["threshold"]
                    )["recall"]
                    if subset.target.nunique() == 2
                    else np.nan,
                }
            )
    pd.DataFrame(groups).to_csv(report_dir / "segment_audit.csv", index=False)
    # SHAP is computed for the SQL CatBoost candidate even if another model wins.
    shap_x = prepare_features(test, medians)[FEATURES]
    shap_values = bundle["estimators"]["catboost_sql"].get_feature_importance(
        Pool(shap_x, cat_features=CATEGORICAL), type="ShapValues", thread_count=threads
    )
    np.savez_compressed(
        report_dir / "shap_values.npz", values=shap_values, feature_names=np.array(FEATURES)
    )
    pd.DataFrame(
        {"feature": FEATURES, "mean_abs_shap": np.abs(shap_values[:, :-1]).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False).to_csv(
        report_dir / "shap_importance.csv", index=False
    )
    bundle_path = artifact_dir / "bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    packages = {
        name: importlib.metadata.version(name)
        for name in [
            "pandas",
            "numpy",
            "scikit-learn",
            "catboost",
            "duckdb",
            "optuna",
            "streamlit",
            "plotly",
        ]
    }
    manifest = {
        "author": "Ainur Ali",
        "seed": seed,
        "champion": champion,
        "selection_rule": "Highest policy-split average precision; test never used for selection",
        "primary_metric": "average_precision",
        "split_protocol": "60% train/CV; 10% calibration; 10% policy; 20% test",
        "best_catboost_params": study.best_params,
        "catboost_cv_ap": study.best_value,
        "best_logistic_C": best_c,
        "cv_folds": 3,
        "optuna_trials": trials,
        "bootstrap_repetitions": bootstrap,
        "default_campaign_fraction": 0.1,
        "bundle_sha256": sha256(bundle_path),
        "source_sha256": sha256(ROOT / "data/raw/telco.csv"),
        "source_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "src/churn_lab").glob("*.py"))
            + sorted((ROOT / "sql").glob("*.sql"))
        },
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.system(),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Champion chosen on policy data: {champion}", flush=True)
    print(
        metrics_table[["model", "roc_auc", "average_precision", "brier"]].to_string(index=False),
        flush=True,
    )
    return manifest
