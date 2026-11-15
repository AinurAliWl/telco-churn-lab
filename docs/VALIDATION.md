# Validation

## Completed checks

- Full experiment executed on Python 3.12.14 / Linux using the pinned dependencies.
- Source CSV verified against SHA-256; 7,043 unique customers and 11 blank `TotalCharges` values accounted for.
- Three-fold training-only model search completed: 16 Optuna trials plus the Logistic Regression reference search.
- A second full run with the same seed reproduced the holdout predictions and SHAP values exactly.
- **42 pytest checks passed**, covering contracts, target-independent SQL features, training/inference consistency, disjoint splits, ranking arithmetic, campaign assumptions, artifact consistency and Streamlit widget flows.
- Saved fitted models reproduced their published holdout probabilities.
- SHAP contributions summed to the CatBoost + SQL raw predictions within numerical tolerance.
- Command-line batch scoring completed for the ten-customer example.
- The case-study notebook's seven code cells executed successfully in an isolated IPython process; outputs are included. The helper in `scripts/execute_case_study.py` reproduces this without a networked kernel.
- Ruff lint and formatting checks passed.
- Generated report figures were rendered and visually inspected.
- Windows-compatible wheels were found for all pinned runtime dependencies. This is not a Windows execution test.

## Application checks

Streamlit's AppTest harness verified initial rendering, model selection, customer selection, zero contact capacity, value-based prioritization and the zero-rescue scenario. No application exceptions occurred in these flows.

These checks exercise application logic and widgets. They do not verify desktop or mobile visual layout, browser downloads or a native browser file-upload interaction.

## Pending checks

- Native Windows execution, including the complete local installation.
- Desktop/mobile visual review and end-to-end browser download/upload checks. Headless Chromium could not start in the validation environment; no successful browser layout test is claimed.
- GitHub Actions execution after pushing the repository. The workflow is configured for Ubuntu and Windows / Python 3.12 but has not run on GitHub.
- A native Jupyter-kernel session. Notebook code execution and output generation were verified through IPython; the kernel transport could not start in the validation environment.

No external telecom validation, production deployment or real retention experiment has been performed.
