# Standard Metrics Library

A lightweight, robust Python library for computing evaluation metrics in machine learning tasks. Focuses on regression and classification (binary/multiclass unified), with optional distributional insights via error quantiles. Built on scikit-learn for reliability.

**NOTE:** Always use the functions in this library for measurement in `sfd.reference_architecture` model functions. 

## Key Features
- **Regression Metrics**: MAE, RMSE, R², mean error, MAPE (with zero-handling).
- **Classification Metrics**: Macro-averaged precision, recall, F1, accuracy, AUC (safe OVR/macro), macro FPR.
- **Unified API**: Single function per task type; supports binary/multiclass seamlessly.
- **Optional Quantiles**: Add error distribution summaries (0.01, 0.25, 0.5, 0.75, 0.99) for deeper analysis.
- **Robustness**: Handles edge cases (e.g., zero divisions, single-class data) with fallbacks.
- **Dependencies**: scikit-learn, numpy (minimal footprint).

## API Reference

### `regression_metrics(data, y_pred, include_quantiles=False)`
- **Args**:
  - `data` (dict): Contains `'y_test'` (array-like).
  - `y_pred` (array-like): Predicted values.
  - `include_quantiles` (bool): Include absolute error quantiles.
- **Returns**: Dict of metrics (rounded to 4 decimals).
- **Notes**: MAPE skips zeros, reports skipped count. Experimental: Validate on your data.

### `classification_metrics(data, y_pred, y_proba=None, include_quantiles=False)`
- **Args**:
  - `data` (dict): Contains `'y_test'` (array-like, integer labels).
  - `y_pred` (array-like): Predicted labels.
  - `y_proba` (array-like, optional): Probabilities (n_samples, n_classes) for AUC.
  - `include_quantiles` (bool): Include absolute error quantiles (label distances).
- **Returns**: Dict of macro-averaged metrics (rounded to 4 decimals).
- **Notes**: AUC falls back to 0.0 if invalid/missing. Macro FPR generalizes binary FPR.

## Examples
- **Research Insight**: Use quantiles to detect error tails in imbalanced datasets—e.g., high q99 flags outliers for further investigation.
- **Engineering Workflow**: Integrate into pipelines for quick validation; e.g., monitor macro FPR in production for misclassification trends.
