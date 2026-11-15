# Model card

**Project:** Telco Churn Lab  
**Author:** Ainur Ali  
**Intended use:** educational churn classification, reproducible model evaluation and illustrative campaign planning.

## Data and target

IBM's fictional Telco sample contains a single customer snapshot with a binary `Churn` label. Available fields include tenure, charges, services, payment method and contract. There is no decision timestamp or validated future prediction horizon. This model is not presented as a next-month forecast.

The exact source is pinned by hash. Missing numeric values are preserved until train-only imputation. Customer ID, target and demographic fields are excluded from the model predictors. Demographics remain available for descriptive segment audits; exclusion does not guarantee fairness.

## Development

Stratified split: 60% training, 10% calibration, 10% policy selection, 20% test. Model search uses three-fold CV inside training. Sigmoid calibration is fitted independently. The policy partition selects calibration, a secondary F1 threshold and the champion by average precision.

The saved champion is `catboost_base`. The additional SQL-feature candidate remains available for comparison and SHAP explanations. See `artifacts/manifest.json` for parameters, dependency versions and checksums.

## Performance

See `reports/RESULTS.md`, `reports/test_metrics.csv` and `reports/uncertainty.json`. The report includes discrimination, probability reliability, capacity-constrained ranking and fixed-model bootstrap intervals. These results apply to a held-out sample drawn from the same illustrative snapshot.

## Limitations

- No temporal validation, external telecom validation or production monitoring history.
- No treatment assignments or counterfactual outcomes; no measured uplift, retention effect or ROI.
- Monthly charge is not customer lifetime value or profit. The planner uses explicit margin and horizon assumptions.
- Small, imbalanced segments can produce unstable metrics. Segment audit results are descriptive and not a fairness certification.
- SHAP explains model associations in raw log-odds; it does not explain causes or effects of proposed offers.
- Fixed-model bootstrap intervals exclude training instability and future distribution shift.
- The sample schema is enforced for scoring; unfamiliar category values require reviewed schema/model changes.
- Bundled Python model artifacts should only be loaded from a trusted source. The app does not accept uploaded serialized models.

## Before operational use

Develop timestamped observation/label windows, validate all features at the decision time, test on later cohorts and the target customer population, measure real intervention effects with an experiment, and define monitoring and human review appropriate to the actual deployment.
