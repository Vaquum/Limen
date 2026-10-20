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
