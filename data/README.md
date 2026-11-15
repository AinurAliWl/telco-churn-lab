# Data provenance

The included CSV is the IBM Telco Customer Churn sample. Each row represents one customer, with account, service and billing attributes and a `Churn` label.

- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- Repository commit: `d5371f5d83a446ad5673cbcca3b814b926491f8a`
- Source file: `data/Telco-Customer-Churn.csv`
- Included copy: `raw/telco.csv`
- Shape: 7,043 rows × 21 columns
- Bytes: 970,457
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- Upstream license: Apache-2.0, copied as `DATA_LICENSE.txt`

The loader verifies the checksum before every rebuild and refuses a changed file. The included source is sufficient for offline execution after installing dependencies. If the file is removed, the loader downloads it from the IBM repository and verifies the same hash.

The data describes a fictional telecom company. It is neither Beeline data nor evidence of a deployable model for Kazakhstan's telecom market.

`TotalCharges` contains 11 blank values. They remain missing in SQL; the final inference transform fills numeric missing values using training-only medians. Logistic imputation/scaling is fitted inside training CV folds. IDs and demographic fields are not model predictors.

There are no observation timestamps, event logs or campaign outcomes. No usage history, treatment labels or future outcomes are fabricated. A stratified customer split is a demonstration protocol, not a temporal forecast validation.

`example_customers.csv` contains ten held-out customers from this same sample, with `Churn` removed. It is a schema example and not an independent external validation set.
