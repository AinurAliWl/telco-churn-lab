# Proposed retention experiment

This is a proposed experiment, not a completed campaign or a measured uplift result.

## Question

Does a retention intervention improve retained contribution after all campaign costs, compared with business-as-usual, among eligible at-risk customers?

## Population and eligibility

- Score currently active customers using only information available at the decision date.
- Freeze eligibility, exclusion rules, capacity and score thresholds before assignment.
- Avoid repeatedly enrolling the same customer or contacting a control customer through another campaign during the measurement window.
- Capture the decision timestamp, observation cutoff, feature version, model version and consent/eligibility state.

## Assignment

Randomly assign eligible customers to a business-as-usual control group and a treatment group. Stratify assignment by pre-treatment risk band and customer value where appropriate. Use a stable assignment key and analyze customers according to original assignment, including nonresponders.

For a first experiment, test one clearly defined intervention. A later NBO experiment can randomize among eligible offers plus control, with sufficient sample size for each arm. Do not infer the best offer by changing contract features in a churn classifier.

## Outcomes and guardrails

- Primary outcome: mean retained contribution per assigned customer over a predefined horizon, net of actual contact and incentive costs.
- Secondary outcomes: retained-customer rate, time to churn and offer acceptance.
- Guardrails: complaint rate, unwanted-contact rate, excessive discounts and repeat-contact burden.
- Record offer eligibility, assigned arm, delivered intervention, actual cost, acceptance, churn outcome and outcome timestamp.

## Analysis

Estimate intention-to-treat differences with confidence intervals. Define sample size from baseline outcome variance or retention rate, a minimum worthwhile effect and desired power before launch. Predefine the analysis window and multiple-testing policy. Do not stop early based on repeated unadjusted significance checks.

Only after collecting suitable randomized outcomes should individual treatment effects, uplift policies or offer ranking be evaluated. Validate any learned policy on independent experimental data. Churn risk, probability calibration and SHAP explanations cannot substitute for this evidence.
