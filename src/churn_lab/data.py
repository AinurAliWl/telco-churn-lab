"""Verified source data, explicit contracts and reusable SQL feature definitions."""

import hashlib
import json
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
CATEGORICAL = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_band",
    "contract_internet",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "addon_count", "total_charges_missing"]
FEATURES = NUMERIC + CATEGORICAL
ALLOWED = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {"0", "1"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}
for _name in [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]:
    ALLOWED[_name] = {"Yes", "No", "No internet service"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_data(path: Path) -> Path:
    """A changed upstream file fails closed instead of silently changing the experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(DATA_URL, timeout=45) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
            raise ValueError("Source checksum mismatch. Review the upstream dataset before use.")
        path.write_bytes(payload)
    if sha256(path) != DATA_SHA256:
        raise ValueError(f"Checksum mismatch: {path}. Restore the original IBM CSV.")
    return path


def validate_raw(frame: pd.DataFrame, *, training: bool = True) -> pd.DataFrame:
    required = RAW_COLUMNS if training else [c for c in RAW_COLUMNS if c != "Churn"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The input CSV contains no customer rows.")
    if len(frame) > 100_000:
        raise ValueError("The batch limit is 100,000 customers.")
    clean = frame[required].copy().fillna("").astype(str)
    for col in required:
        clean[col] = clean[col].str.strip()
    if clean.customerID.eq("").any() or clean.customerID.duplicated().any():
        raise ValueError("customerID must be non-empty and unique.")
    for col, allowed in ALLOWED.items():
        unexpected = set(clean[col]) - allowed
        if unexpected:
            raise ValueError(f"Unexpected values in {col}: {sorted(unexpected)[:5]}")
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        numeric = pd.to_numeric(clean[col], errors="coerce")
        valid_blank = (
            clean[col].eq("") if col == "TotalCharges" else pd.Series(False, index=clean.index)
        )
        invalid = (~np.isfinite(numeric) | (numeric < 0)) & ~valid_blank
        if invalid.any():
            raise ValueError(f"{col} must contain finite, non-negative numbers.")
        if col == "tenure" and ((numeric % 1 != 0) | (numeric > 1200)).any():
            raise ValueError("tenure must be an integer between 0 and 1200 months.")
    if training and not set(clean.Churn).issubset({"Yes", "No"}):
        raise ValueError("Churn must contain only Yes / No.")
    return clean


def feature_frame(raw: pd.DataFrame, *, training: bool = True) -> pd.DataFrame:
    """The same row-wise SQL runs for training and batch inference. No target aggregation."""
    clean = validate_raw(raw, training=training)
    if not training:
        clean["Churn"] = "No"  # Placeholder for SQL output only; never included in FEATURES.
    with duckdb.connect() as con:
        con.register("customers_raw", clean)
        result = con.execute((ROOT / "sql/features.sql").read_text()).df()
    for col in CATEGORICAL:
        result[col] = result[col].astype(str)
    if not training:
        result = result.drop(columns="target")
    return result


def make_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """60% fit/CV, 10% calibration, 10% policy selection, 20% untouched test."""
    ids = np.arange(len(frame))
    remaining, test = train_test_split(ids, test_size=0.2, stratify=frame.target, random_state=seed)
    train, reserve = train_test_split(
        remaining, test_size=0.25, stratify=frame.target.iloc[remaining], random_state=seed + 1
    )
    calibration, policy = train_test_split(
        reserve, test_size=0.5, stratify=frame.target.iloc[reserve], random_state=seed + 2
    )
    result = pd.DataFrame({"customerID": frame.customerID, "split": ""})
    for name, idx in [
        ("train", train),
        ("calibration", calibration),
        ("policy", policy),
        ("test", test),
    ]:
        result.loc[idx, "split"] = name
    assert len(result) == len(frame) and result.split.ne("").all()
    return result


def build_data(out: Path, seed: int = 42) -> pd.DataFrame:
    source = download_data(ROOT / "data/raw/telco.csv")
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    features = feature_frame(raw)
    splits = make_splits(features, seed)
    result = features.merge(splits, on="customerID", validate="one_to_one")
    out.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out / "features.parquet", index=False)
    splits.to_csv(out / "splits.csv", index=False)
    with duckdb.connect(str(out / "analytics.duckdb")) as con:
        con.register("prepared", result)
        con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM prepared")
        segments = con.execute((ROOT / "sql/segments.sql").read_text()).df()
    segments.to_csv(out / "segments_train.csv", index=False)
    quality = {
        "rows": len(result),
        "source_sha256": sha256(source),
        "unique_customers": int(raw.customerID.nunique()),
        "missing_total_charges": int(raw.TotalCharges.str.strip().eq("").sum()),
        "features": FEATURES,
        "excluded_demographics": ["gender", "SeniorCitizen", "Partner", "Dependents"],
        "splits": result.groupby("split")
        .agg(rows=("target", "size"), churn_rate=("target", "mean"))
        .round(6)
        .to_dict("index"),
    }
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    return result
