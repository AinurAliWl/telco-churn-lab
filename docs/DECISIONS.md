# Design decisions

## A decision problem, not only a classifier

I framed the project around contact capacity. A retention team cannot contact every customer, so ranking quality at 5%, 10% and 20% capacity matters alongside global discrimination. The app separates observed retrospective coverage from assumed future campaign value.

## DuckDB for inspectable SQL

Feature transformations and segment queries live in versioned SQL files. DuckDB runs locally without a database server. Training and CSV scoring call the same feature query, avoiding separate definitions for notebook and application inputs. Learned transforms remain in model fitting: no cohort statistic is computed before splitting.

## A linear baseline and a controlled boosting comparison

Logistic Regression provides a strong reference. CatBoost handles the categorical service/account data without a separate one-hot preprocessing step. Optuna uses three-fold training-only average precision to search a bounded space of depth, learning rate, regularization and iterations.

The SQL-feature variant adds service count, tenure band, a contract/internet interaction and a missing-total-charges flag. The base-feature CatBoost reuses its selected hyperparameters. This is a fixed-configuration ablation rather than an independently tuned comparison.

The policy split selected the base-feature CatBoost. Extra feature engineering was retained as an experiment, not advertised as a guaranteed improvement.

## Four disjoint partitions

A separate calibration partition avoids fitting the probability correction on the same customers used to fit the classifier. A further policy partition chooses calibration, model and threshold. The test partition is reserved for final reporting. This uses fewer training customers than a simple split; it makes the responsibilities explicit for a compact educational experiment.

## Calibration is evaluated, not assumed to help

A sigmoid logistic correction is fitted to raw logit scores. The policy split selects calibrated output only when it improves Brier score and preserves the score direction. Raw and calibrated outputs remain available in the reports. Model selection uses AP; the secondary threshold maximizes F1 on policy data. F1 is not claimed to optimize business profit.

In the published run, calibration increased policy-set Brier score for all candidates. All three therefore retained raw output probabilities. The app still exposes reliability plots and the saved reports include the calibrated alternatives.

## SHAP explanations stay attached to their model

The explanation view always uses the CatBoost + SQL candidate. It labels the model explicitly, even if the selected champion differs. Contributions add to its raw log-odds, not calibrated probability. Changing an input feature is not evidence of a causal intervention effect.

## Value-first planning is a scenario

Ranking by expected saved contribution can differ from ranking by churn risk. The planner makes costs and a constant conditional rescue rate explicit. It does not learn who is persuadable. Measuring that requires treatment/control outcomes; the experiment plan describes how to obtain them.

## Keep the demo self-contained

Saved artifacts and reports allow the app to run without training or API access. The CLI supports reproduction and batch inference. No cloud services, LLM, feature store server or orchestration cluster is required for this small snapshot experiment.
