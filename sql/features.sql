-- Row-local transforms only. Reused unchanged during training and inference.
-- No labels, cohort means or future information enter the model's feature list.
SELECT
    customerID, gender, SeniorCitizen, Partner, Dependents,
    CAST(tenure AS DOUBLE) AS tenure,
    CAST(MonthlyCharges AS DOUBLE) AS MonthlyCharges,
    TRY_CAST(NULLIF(TotalCharges, '') AS DOUBLE) AS TotalCharges,
    PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    CAST(OnlineSecurity = 'Yes' AS INTEGER)
        + CAST(OnlineBackup = 'Yes' AS INTEGER)
        + CAST(DeviceProtection = 'Yes' AS INTEGER)
        + CAST(TechSupport = 'Yes' AS INTEGER)
        + CAST(StreamingTV = 'Yes' AS INTEGER)
        + CAST(StreamingMovies = 'Yes' AS INTEGER) AS addon_count,
    CASE
        WHEN CAST(tenure AS DOUBLE) <= 12 THEN '00-12 months'
        WHEN CAST(tenure AS DOUBLE) <= 24 THEN '13-24 months'
        WHEN CAST(tenure AS DOUBLE) <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_band,
    Contract || ' / ' || InternetService AS contract_internet,
    CAST(NULLIF(TotalCharges, '') IS NULL AS INTEGER) AS total_charges_missing,
    CAST(Churn = 'Yes' AS INTEGER) AS target
FROM customers_raw
