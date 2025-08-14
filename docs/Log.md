# Log

## `loop.log`

### `Log`

Create Log object state from a UEL object or a log file.

#### Args

| Parameter             | Type          | Description                           |
|-----------------------|---------------|---------------------------------------|
| `uel_object`          | `object, optional`   | Source UEL object                      |
| `file_path`           | `str, optional`      | Path to the log file                   |
| `inverse_scaler`      | `Callable, optional` | Inverse scaler function                |
| `cols_to_multilabel`  | `list[str], optional`| Columns to convert to multilabel       |

### `experiment_backtest_results`

Compute backtest performance metrics for each round of an experiment. It shows how often your trades win, how much they make on average, how deep the worst drawdowns are, and whether the strategy delivers consistent returns after costs. This lets you see not just if your system works, but how it performs under realistic trading conditions.

#### Args

| Parameter               | Type   | Description                         |
|-------------------------|--------|-------------------------------------|
| `disable_progress_bar`  | `bool` | Whether to disable the progress bar |

#### Returns

`pd.DataFrame`: One-row-per-round table with the following columns.

| Column                      | Type     | Brief description (1-line)                                   |
|-----------------------------|----------|---------------------------------------------------------------|
| `trade_win_rate_pct`        | float64  | Percentage of trades with positive net return                 |
| `trade_expectancy_pct`      | float64  | Mean net return per in-market bar (percent units)            |
| `max_drawdown_pct`          | float64  | Maximum peak-to-trough drawdown from net equity curve        |
| `total_return_gross_pct`    | float64  | Cumulative gross return across all bars (percent)            |
| `total_return_net_pct`      | float64  | Cumulative net return including round-trip costs (percent)   |
| `trade_return_mean_win_pct` | float64  | Mean net return of winning bars (percent)                    |
| `trade_return_mean_loss_pct`| float64  | Mean net return of losing bars (percent)                     |
| `bars_total`                | int64    | Total number of evaluated bars                               |
| `sharpe_per_bar`            | float64  | Mean/SD of net per-bar returns                               |
| `bars_in_market_pct`        | float64  | Percentage of bars with an active long position              |
| `trades_count`              | int64    | Count of trades (per selected counting mode)                 |
| `cost_round_trip_bps`       | int64    | Effective bps charged per round trip (2 × (fee + slippage))  |

### `experiment_confusion_metrics`

Compute confusion metrics for each round of an experiment. One way to think about it is that it tells you if your LONG calls are mostly right (precision), if you catch most of the real LONG cases (recall), how common LONG is, and whether the “wins” (TP) actually look better on your chosen metric x than the “fake wins” (FP). If TP and FP are well separated on x (big Cohen’s d / KS), your LONG signal is not just correct more often—it’s also meaningfully profitable/powerful on the thing you care about.

#### Args

| Parameter               | Type   | Description                                       |
|-------------------------|--------|---------------------------------------------------|
| `x`                     | `str`  | Column name to compute confusion metrics for      |
| `disable_progress_bar`  | `bool` | Whether to disable the progress bar               |

#### Returns

`pd.DataFrame`: One-row-per-round table with confusion and long-only summary with the following columns.

| Column                   | Type     | Brief description (1-line)                                          |
|--------------------------|----------|---------------------------------------------------------------------|
| `x_name`                 | object   | Name of the summarized column                                       |
| `n_kept`                 | int64    | Rows kept after outlier handling                                    |
| `pred_pos_rate_pct`      | float64  | Share of predicted positives                                        |
| `actual_pos_rate_pct`    | float64  | Share of actual positives                                           |
| `precision_pct`          | float64  | TP / (TP + FP)                                                      |
| `recall_pct`             | float64  | TP / (TP + FN)                                                      |
| `pred_pos_count`         | int64    | Count of predicted positives                                        |
| `tp_count`               | int64    | True positive count                                                 |
| `fp_count`               | int64    | False positive count                                                |
| `tp_x_mean`              | float64  | Mean of `x` within TP rows                                          |
| `tp_x_median`            | float64  | Median of `x` within TP rows                                        |
| `fp_x_mean`              | float64  | Mean of `x` within FP rows                                          |
| `fp_x_median`            | float64  | Median of `x` within FP rows                                        |
| `pred_pos_x_mean`        | float64  | Mean of `x` within predicted positives                              |
| `pred_pos_x_median`      | float64  | Median of `x` within predicted positives                            |
| `tp_fp_cohen_d`          | float64  | Cohen's d effect size between TP and FP `x` distributions           |
| `tp_fp_ks`               | float64  | KS statistic between TP and FP `x` distributions                    |

NOTE: Additional identifier columns from round parameters may be included.

### `experiment_feature_correlation`

Compute robust correlations between numeric features and a target metric across explicit cohorts. It measures how strongly each feature moves with your chosen metric (e.g., auc) within specific slices of the data, using bootstrapping to give stable estimates and confidence intervals. It helps identify features that consistently align with high or low metric values, and how stable those relationships are across the data distribution.

#### Args

| Parameter          | Type                                | Description                                                 |
|--------------------|-------------------------------------|-------------------------------------------------------------|
| `metric`           | `str`                               | Target column to correlate against (e.g., 'auc')            |
| `sort_key`         | `str \| None`                       | Column used to rank rows before slicing                     |
| `sort_ascending`   | `bool`                              | Whether to sort ascending when creating cohorts             |
| `heads`            | `Sequence[float \| int] \| None`    | Cohort sizes; fractions (0, 1] as proportions or integers as counts |
| `method`           | `str`                               | Correlation method: 'spearman', 'pearson', or 'kendall'     |
| `n_boot`           | `int`                               | Number of bootstrap resamples per cohort                    |
| `min_n`            | `int`                               | Minimum cohort size; smaller cohorts are skipped            |
| `random_state`     | `int`                               | RNG seed for reproducibility                                |

#### Returns

`pd.DataFrame`: MultiIndex rows indexed by ('cohort_pct', 'feature') with the following columns.

| Column            | Type     | Brief description (1-line)                           |
|-------------------|----------|------------------------------------------------------|
| `n_rows`          | int64    | Number of rows in the cohort                         |
| `corr`            | float64  | Correlation in the cohort                            |
| `corr_med`        | float64  | Bootstrap median correlation                         |
| `ci_lo`           | float64  | 2.5% bootstrap quantile of correlation               |
| `ci_hi`           | float64  | 97.5% bootstrap quantile of correlation              |
| `sign_stability`  | float64  | Share of bootstrap samples with the median's sign    |

NOTE: Non-numeric columns are coerced with errors='coerce' and ignored thereafter; constant or all-NaN numeric columns are dropped; rows with NaN in `metric` are dropped prior to sorting and slicing

### `permutation_confusion_metrics`

Compute confusion metrics for a single round of an experiment. One way to think about it is that it tells you if your LONG calls are mostly right (precision), if you catch most of the real LONG cases (recall), how common LONG is, and whether the “wins” (TP) actually look better on your chosen metric x than the “fake wins” (FP). If TP and FP are well separated on x (big Cohen’s d / KS), your LONG signal is not just correct more often—it’s also meaningfully profitable/powerful on the thing you care about.

#### Args

| Parameter           | Type                    | Description                                                              |
|---------------------|-------------------------|--------------------------------------------------------------------------|
| `x`                 | `str`                   | Column summarized within TP/FP/TN/FN (e.g., predicted_probability or P&L)|
| `pred_col`          | `str`                   | Binary predictions column                                                |
| `actual_col`        | `str`                   | Binary actuals column                                                    |
| `proba_col`         | `str \| None`           | Probabilities to binarize via `threshold` (overrides `pred_col`)         |
| `threshold`         | `float`                 | Decision threshold for `proba_col`                                       |
| `outlier_quantiles` | `Sequence[float]`       | (lo, hi) for x outlier handling                                          |
| `outlier_mode`      | `str`                   | 'filter' to drop outside bounds or 'winsor' to clip                      |
| `id_cols`           | `dict[str, Any] \| None`| Optional identifiers to prepend (e.g., params)                           |

#### Returns

`pd.DataFrame`: One-row table with the following columns.

| Column                | Type     | Brief description (1-line)                                          |
|-----------------------|----------|---------------------------------------------------------------------|
| `x_name`              | object   | Name of the summarized column                                       |
| `n_kept`              | int64    | Rows kept after outlier handling                                    |
| `pred_pos_rate_pct`   | float64  | Share of predicted positives                                        |
| `actual_pos_rate_pct` | float64  | Share of actual positives                                           |
| `precision_pct`       | float64  | TP / (TP + FP)                                                      |
| `recall_pct`          | float64  | TP / (TP + FN)                                                      |
| `pred_pos_count`      | int64    | Count of predicted positives                                        |
| `tp_count`            | int64    | True positive count                                                 |
| `fp_count`            | int64    | False positive count                                                |
| `tp_x_mean`           | float64  | Mean of `x` within TP rows                                          |
| `tp_x_median`         | float64  | Median of `x` within TP rows                                        |
| `fp_x_mean`           | float64  | Mean of `x` within FP rows                                          |
| `fp_x_median`         | float64  | Median of `x` within FP rows                                        |
| `pred_pos_x_mean`     | float64  | Mean of `x` within predicted positives                              |
| `pred_pos_x_median`   | float64  | Median of `x` within predicted positives                            |
| `tp_fp_cohen_d`       | float64  | Cohen's d effect size between TP and FP `x` distributions           |
| `tp_fp_ks`            | float64  | KS statistic between TP and FP `x` distributions                    |

### `permutation_prediction_performance`

Create a table of model predictions, actual outcomes, and basic price movement stats for a given experiment round, bar-by-bar. This provides the raw hit/miss information and price data that other evaluation functions (e.g., confusion metrics, backtests) build on.

#### Args

| Parameter  | Type  | Description                                        |
|------------|-------|----------------------------------------------------|
| `round_id` | `int` | Round ID (i.e. nth permutation in an experiment)   |

#### Returns

`pd.DataFrame`: Table with the following columns.

| Column         | Type     | Brief description (1-line)                |
|----------------|----------|-------------------------------------------|
| `predictions`  | int64    | Binary model prediction                   |
| `actuals`      | int64    | Binary ground truth                       |
| `hit`          | bool     | 1 if prediction equals actual             |
| `miss`         | bool     | 1 if prediction differs from actual       |
| `open`         | float64  | Open price for the bar                    |
| `close`        | float64  | Close price for the bar                   |
| `price_change` | float64  | `close - open`                            |

### `_get_test_data_with_all_cols`

Compute test-period rows with all columns.

#### Args

| Parameter   | Type  | Description |
|-------------|-------|-------------|
| `round_id`  | `int` | Round ID    |

#### Returns

`pl.DataFrame`: Klines dataset filtered down to the permutation test window

### `read_from_file`

Create cleaned experiment log DataFrame from file.

#### Args

| Parameter    | Type   | Description                          |
|--------------|--------|--------------------------------------|
| `file_path`  | `str`  | Path to experiment log CSV file      |

#### Returns

`pd.DataFrame`: Cleaned log data with whitespace-trimmed object columns
