-- Exploration is restricted to the training split.
SELECT
    Contract,
    InternetService,
    COUNT(*) AS customers,
    SUM(target) AS churned,
    AVG(target) AS churn_rate,
    AVG(MonthlyCharges) AS mean_monthly_charge,
    AVG(tenure) AS mean_tenure
FROM customers
WHERE split = 'train'
GROUP BY Contract, InternetService
ORDER BY churn_rate DESC
