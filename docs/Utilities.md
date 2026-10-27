# Utilities

Utilities are the smaller helper surfaces that support Limen workflows without defining one primary subsystem of their own.

The [Python project](examples/python-research/README.md) keeps its chronological preparation, fitted scaler and validation score in caller-owned functions. The [Compiled project](examples/compiled-research/README.md) obtains prepared split dictionaries from its Manifest and uses reference architectures that convert them for model fitting. These utilities support those boundaries; they do not define another research interface or replace the projects' complete workflows.

## Current public utility surface

| Helper | Use case |
|---|---|
| `ParamSpace` | legacy standard UEL path requiring sampled parameter combinations |
| `data_dict_to_numpy` | numpy arrays from the standard Limen `data_dict` |
| `adf_test` and `AdfResult` | stationarity check for a series or for helpers such as `find_min_d` |
| `confidence_filtering_system` | validation-calibrated confidence filtering across multiple models |
| `split_by_dates` | absolute-datetime train / val / test splits (the helper backing `Manifest.set_split_dates`) |
| reporting helpers | formatted text blocks |

## `ParamSpace`

`ParamSpace` is the legacy permutation helper used by the standard non-MSQ run path.

<!-- reference: utilities-paramspace-constructor source=limen/utils/param_space.py region=signature:ParamSpace.__init__ role=signature -->
```python
def __init__(self, params: dict[str, list[Any]], n_permutations: int, seed: int | None=None, *, sample: bool=True) -> None:
    ...
```

The constructor distinguishes a sampled pool from deterministic prefix enumeration. This is an API reference; neither canonical project uses the legacy `ParamSpace` path.

By default, `generate(random_search=False)` consumes the sampled pool in order. Construct with `sample=False` to select the first `n_permutations` combinations from the declared grid; then `generate(random_search=False)` enumerates that prefix, with the first parameter varying fastest. Standard `UniversalExperimentLoop.run(random_search=False)` selects and consumes this deterministic prefix.

`ParamSpace` uses an instance-local random generator. Pass `seed=` for reproducible legacy sampling; it does not read or mutate Python's module-global random state. Standard `UniversalExperimentLoop` construction does not expose that seed, so instantiate `ParamSpace` directly when you need seeded legacy helper sampling. This legacy helper is outside the artifact-backed `round_params` reproducibility contract.

`ParamSpace` supports the [legacy execution contract](Universal-Experiment-Loop.md#legacy-execution-reference). For new Python experiments, use a strategy through the [Python/MSQ route](Universal-Experiment-Loop.md#direct-python-run).

## `data_dict_to_numpy`

`data_dict_to_numpy()` converts the standard split keys from polars or pandas into numpy arrays.

<!-- reference: utilities-numpy-conversion source=limen/utils/data_dict_to_numpy.py region=signature:data_dict_to_numpy role=signature -->
```python
def data_dict_to_numpy(data: dict[str, Any], keys: list[str] | None=None) -> dict[str, Any]:
    ...
```

The first dimension follows each split's row count; feature arrays retain the prepared feature width.

This helper supports sklearn-style or numpy-first model code.

`split_data_to_prep_output()` is the companion converter that builds `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test` from split frames without mutating the passed split list or column list.

## `adf_test`

`adf_test()` runs an Augmented Dickey-Fuller stationarity test and returns an `AdfResult`.

Install the statistical extra first: `pip install "vaquum-limen[stats]"`. The helper imports `statsmodels` lazily, so the rest of `limen.utils` remains available without it.

<!-- reference: utilities-adf source=limen/utils/adf_test.py region=signature:adf_test role=signature -->
```python
def adf_test(series: pl.Series, significance_level: float=0.05) -> AdfResult:
    ...
```

<!-- reference: utilities-adf-result source=limen/utils/adf_test.py region=signature:AdfResult role=signature -->
```python
@dataclass
class AdfResult:
    stationary: bool
    p_value: float
    test_statistic: float
    critical_values: dict[str, float]
```

The structured result contains:

- `stationary`
- `p_value`
- `test_statistic`
- `critical_values`

This is the utility layer that [Features](Features.md) now uses for `find_min_d()`.

## `confidence_filtering_system`

`confidence_filtering_system()` is a higher-level utility for post-prediction filtering based on agreement across multiple models.

It expects a data dictionary containing at least:

- `x_val`, `y_val`
- `x_test`, `y_test`
- `dt_test`

The utility validates this contract before computing metrics: `target_confidence` must be finite and in `[0.0, 1.0]`, targets must be finite one-dimensional numeric arrays, each model must expose `predict()`, and every model prediction vector must be finite, one-dimensional, and match the target length.

It returns:

1. a results dictionary
2. a detailed polars results frame
3. calibration statistics

The detailed results frame describes each test observation with `datetime`, `prediction`, `uncertainty`, `is_confident`, `confidence_threshold`, `actual_value` and `confidence_score`. Coverage depends on the caller's validation population, target confidence and model agreement; this reference supplies no invented coverage result.

<!-- reference: utilities-confidence-filtering source=limen/utils/confidence_filtering_system.py region=signature:confidence_filtering_system role=signature -->
```python
def confidence_filtering_system(models: list[Any], data: dict[str, Any], target_confidence: float=0.8) -> tuple[dict[str, Any], pl.DataFrame, dict[str, Any]]:
    ...
```

Use this as an optional downstream helper, not as part of the core UEL contract.

## `split_by_dates`

`split_by_dates()` partitions a datetime-indexed `polars.DataFrame` into train, val, and test by half-open `[start, end)` datetime windows. It is the helper that `Manifest.set_split_dates` calls when `split_dates` is configured; it is also exported for direct use when the splitter is needed outside the manifest pipeline.

<!-- reference: utilities-date-splits source=limen/data/utils/splits.py region=signature:split_by_dates role=signature -->
```python
def split_by_dates(data: pl.DataFrame, train_start: date | Any, train_end: date | Any, val_start: date | Any, val_end: date | Any, test_start: date | Any, test_end: date | Any) -> list[pl.DataFrame]:
    ...
```

Behavior rules:

- Each window selects its rows independently. No row from outside all three windows enters any split.
- When windows are non-overlapping (the ordering contract `set_split_dates` enforces), no row appears in more than one split.
- Gaps between adjacent windows are allowed; rows that fall inside a gap are intentionally excluded from all three splits.
- Every bound must be a `date` or `datetime` instance. Strings, ints, and floats raise `TypeError` at the API boundary.
- The input `DataFrame` must have a `datetime` column.

Prefer `Manifest.set_split_dates` inside the experiment pipeline — it pairs the splitter with the manifest's ordering validation and `with_params_override` clearance contract. Use `split_by_dates` directly when the manifest pipeline is not in play.

## Reporting helpers

The reporting helpers are small text-formatting functions:

- `format_report_header`
- `format_report_section`
- `format_report_footer`

These are text-formatting utilities, not a canonical reporting framework.

## Read next

- Continue to the [legacy execution reference](Universal-Experiment-Loop.md#legacy-execution-reference) for the contract that still uses `ParamSpace`.
- Continue to [Advanced Search](Advanced-Search.md) for the strategies used by CLI and Python/MSQ runs.
- Continue to [Reference Architecture](Reference-Architecture.md) for `data_dict_to_numpy()` usage inside model code.

## Callable contracts

These entries give the exact public signatures and the source-owned argument, return and failure contracts. Use the project examples above for the research context; a signature is reference evidence, not a claim that every option ran in the base experiment.

### `limen.utils.AdfResult`

<!-- reference: api-limen-utils-adfresult source=limen/utils/adf_test.py region=signature:AdfResult role=signature -->
```python
@dataclass
class AdfResult:
    stationary: bool
    p_value: float
    test_statistic: float
    critical_values: dict[str, float]
```

Result of an Augmented Dickey-Fuller stationarity test.

### `limen.utils.ParamSpace`

<!-- reference: api-limen-utils-paramspace source=limen/utils/param_space.py region=signature:ParamSpace role=signature -->
```python
class ParamSpace:
    ...
```

Sample parameter combinations without constructing the Cartesian product.

Preserve stdlib sampling beyond Py_ssize_t; sample=False selects the grid prefix.

### `limen.utils.ParamSpace.__init__`

<!-- reference: api-limen-utils-paramspace---init-- source=limen/utils/param_space.py region=signature:ParamSpace.__init__ role=signature -->
```python
def __init__(self, params: dict[str, list[Any]], n_permutations: int, seed: int | None=None, *, sample: bool=True) -> None:
    ...
```

### `limen.utils.ParamSpace.generate`

<!-- reference: api-limen-utils-paramspace-generate source=limen/utils/param_space.py region=signature:ParamSpace.generate role=signature -->
```python
def generate(self, random_search: bool=True) -> dict[str, Any] | None:
    ...
```

Compute next parameter combination from the parameter space.

| Argument | Contract |
| --- | --- |
| random_search (`bool`) | Whether to select parameters randomly or sequentially |

**Returns:** dict: Dictionary of parameter names and selected values, or None if space is exhausted

### `limen.utils.adf_test`

<!-- reference: api-limen-utils-adf-test source=limen/utils/adf_test.py region=signature:adf_test role=signature -->
```python
def adf_test(series: pl.Series, significance_level: float=0.05) -> AdfResult:
    ...
```

Run Augmented Dickey-Fuller test for stationarity.

| Argument | Contract |
| --- | --- |
| series (`pl.Series`) | Time series to test |
| significance_level (`float`) | Threshold for rejecting non-stationarity |

**Returns:** AdfResult: Structured test result with stationary flag, p-value,

test statistic, and critical values

### `limen.utils.binary_metrics`

<!-- reference: api-limen-utils-binary-metrics source=limen/metrics/binary_metrics.py region=signature:binary_metrics role=signature -->
```python
def binary_metrics(data: dict[str, Any], preds: list[int] | npt.NDArray[np.integer[Any]], probs: list[float] | npt.NDArray[np.floating[Any]]) -> dict[str, Any]:
    ...
```

Compute binary classification metrics from predictions and probabilities.

Degenerate one-class outcomes never crash: fpr is NaN when y_test has no

negatives, and auc is NaN when y_test contains a single class, mirroring

the safe_ovr_auc convention.

| Argument | Contract |
| --- | --- |
| data (`dict`) | Data dictionary with 'y_test' key containing true binary labels |
| preds (`list`) | Predicted binary class labels |
| probs (`list`) | Predicted class probabilities |

**Returns:** dict: Dictionary containing recall, precision, fpr, auc, and accuracy metrics

### `limen.utils.confidence_filtering_system`

<!-- reference: api-limen-utils-confidence-filtering-system source=limen/utils/confidence_filtering_system.py region=signature:confidence_filtering_system role=signature -->
```python
def confidence_filtering_system(models: list[Any], data: dict[str, Any], target_confidence: float=0.8) -> tuple[dict[str, Any], pl.DataFrame, dict[str, Any]]:
    ...
```

Compute complete confidence filtering system with validation-based calibration.

| Argument | Contract |
| --- | --- |
| models (`list`) | List of trained models for confidence estimation |
| data (`dict`) | Dictionary with validation and test data splits containing 'x_val', 'y_val', 'x_test', 'y_test', 'dt_test' keys |
| target_confidence (`float`) | Target percentage of predictions to classify as confident |

**Returns:** tuple: Filtered test results, per-row results DataFrame sorted by

confidence score, and calibration statistics

### `limen.utils.continuous_metrics`

<!-- reference: api-limen-utils-continuous-metrics source=limen/metrics/continuous_metrics.py region=signature:continuous_metrics role=signature -->
```python
def continuous_metrics(data: dict[str, Any], preds: list[float] | npt.NDArray[np.floating[Any]]) -> dict[str, Any]:
    ...
```

Compute regression metrics from continuous predictions.

NOTE: This function is experimental and may change in future versions.

| Argument | Contract |
| --- | --- |
| data (`dict`) | Data dictionary with 'y_test' key containing true continuous values |
| preds (`list`) | Predicted continuous values |

**Returns:** dict: Dictionary containing bias, mae, rmse, r2, and mape metrics

### `limen.utils.data_dict_to_numpy`

<!-- reference: api-limen-utils-data-dict-to-numpy source=limen/utils/data_dict_to_numpy.py region=signature:data_dict_to_numpy role=signature -->
```python
def data_dict_to_numpy(data: dict[str, Any], keys: list[str] | None=None) -> dict[str, Any]:
    ...
```

Convert data dictionary entries from polars/pandas to numpy arrays.

| Argument | Contract |
| --- | --- |
| data (`dict`) | Dictionary with data entries (e.g., x_train, y_train, etc.) |
| keys (`list`) | Keys to convert. Defaults to x_train, y_train, x_val, y_val, x_test, y_test |

**Returns:** dict: Dictionary with values converted to numpy arrays where applicable

### `limen.utils.filter_lines_by_quantile`

<!-- reference: api-limen-utils-filter-lines-by-quantile source=limen/utils/filter_lines_by_quantile.py region=signature:filter_lines_by_quantile role=signature -->
```python
def filter_lines_by_quantile(lines: list[dict[str, float]], quantile: float) -> list[dict[str, float]]:
    ...
```

Compute the subset of lines at or above a height quantile.

| Argument | Contract |
| --- | --- |
| lines (`list[dict]`) | Line dicts with a 'height_pct' key, as produced by find_price_lines |
| quantile (`float`) | Quantile of absolute heights below which lines are dropped |

**Returns:** list[dict]: Lines whose absolute height is at or above the quantile cutoff

**Raises:** ValueError: If quantile is outside [0, 1]

### `limen.utils.find_price_lines`

<!-- reference: api-limen-utils-find-price-lines source=limen/utils/find_price_lines.py region=signature:find_price_lines role=signature -->
```python
def find_price_lines(close: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]], max_duration_hours: int, min_height_pct: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    ...
```

Compute linear price movements (lines) from a close-price series.

A line is any pair of bars (start, end) with duration in

[1, max_duration_hours) whose relative close-to-close change is at

least min_height_pct in absolute terms. Lines with a positive change

are long lines, negative are short lines.

| Argument | Contract |
| --- | --- |
| close (`np.ndarray`) | One-dimensional array of close prices |
| max_duration_hours (`int`) | Exclusive upper bound on line duration in bars |
| min_height_pct (`float`) | Minimum absolute height as a fraction of the start price |

**Returns:** tuple[list[dict], list[dict]]: (long_lines, short_lines), each a list of

dicts with keys 'start_idx', 'end_idx', 'height_pct', 'duration_hours'

**Raises:** ValueError: If max_duration_hours is below 2 or min_height_pct is not positive

NOTE: Lines are detected only within the array passed in, so per-split

invocation cannot observe other splits.

### `limen.utils.format_report_footer`

<!-- reference: api-limen-utils-format-report-footer source=limen/utils/reporting.py region=signature:format_report_footer role=signature -->
```python
def format_report_footer(width: int=50) -> str:
    ...
```

Create standardized report footer with consistent styling.

| Argument | Contract |
| --- | --- |
| width (`int`) | Width of the separator line |

**Returns:** str: Formatted footer string

### `limen.utils.format_report_header`

<!-- reference: api-limen-utils-format-report-header source=limen/utils/reporting.py region=signature:format_report_header role=signature -->
```python
def format_report_header(title: str, width: int=50) -> str:
    ...
```

Create standardized report header with consistent styling.

| Argument | Contract |
| --- | --- |
| title (`str`) | Title of the report section |
| width (`int`) | Width of the separator line |

**Returns:** str: Formatted header string with separators

### `limen.utils.format_report_section`

<!-- reference: api-limen-utils-format-report-section source=limen/utils/reporting.py region=signature:format_report_section role=signature -->
```python
def format_report_section(title: str, width: int=50) -> str:
    ...
```

Create standardized report section with consistent styling.

| Argument | Contract |
| --- | --- |
| title (`str`) | Title of the report section |
| width (`int`) | Width of the separator line |

**Returns:** str: Formatted section string with separators

### `limen.utils.safe_ovr_auc`

<!-- reference: api-limen-utils-safe-ovr-auc source=limen/metrics/safe_ovr_auc.py region=signature:safe_ovr_auc role=signature -->
```python
def safe_ovr_auc(y_true: npt.NDArray[Any] | list[int] | list[str], probs: npt.NDArray[np.floating[Any]] | list[list[float]]) -> float:
    ...
```

Compute one-vs-rest AUC safely handling missing classes.

| Argument | Contract |
| --- | --- |
| y_true (`np.ndarray \| list[int] \| list[str]`) | True class labels, shape (n_samples,) |
| probs (`np.ndarray \| list[list[float]]`) | Predicted probabilities, shape (n_samples, n_classes) |

**Returns:** float: Mean AUC across all valid class comparisons, or NaN if no valid AUC calculations can be made

## Temporal perturbation selectors

`lookback_window` selects the inclusive trailing elapsed-day interval ending at the last supplied observation. It requires unique, increasing, non-null datetime observations and a positive finite day count.

`sample_regime_blocks` computes the existing causal market classifier under an explicit `regime_version`, identifies complete contiguous runs of `target_regime`, and chooses one uniformly with `seed`. It refuses initializing/unknown regimes, multiple series and populations without a sufficiently long block. It returns original rows without adding regime features or concatenating separated runs. Optional `lookback_days` bounds the classifier input before selection. A selected population may be unsuitable for fixed date splits; normal split admission remains authoritative. Results describe this conditional population.

Both functions can be declared through the existing manifest/YAML `pre_split_data_selector`, with parameter references for their controls.

<!-- reference: utilities-lookback-window source=limen/data/utils/lookback_window.py region=signature:lookback_window role=signature -->
```python
def lookback_window(data: pl.DataFrame, lookback_days: float) -> pl.DataFrame:
    ...
```

<!-- reference: utilities-sample-regime-blocks source=limen/data/utils/sample_regime_blocks.py region=signature:sample_regime_blocks role=signature -->
```python
def sample_regime_blocks(data: pl.DataFrame, target_regime: str, min_block_size: int, *, seed: int, regime_version: str, lookback_days: float | None=None) -> pl.DataFrame:
    ...
```
