# Utilities

Utilities are the smaller helper surfaces that support Limen workflows without defining one primary subsystem of their own.

They support the package, but they are not the main workflow. New Limen readers should start with [Universal Experiment Loop](Universal-Experiment-Loop.md), [Experiment Manifest](Experiment-Manifest.md), and [Log](Log.md).

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

```python
from limen.utils import ParamSpace

ps = ParamSpace(
    {'alpha': [0.1, 0.2], 'beta': ['x', 'y'], 'gamma': [1, 2]},
    n_permutations=3,
    seed=42,
)
```

On a live local run in this repo, that parameter space had:

- `total_space = 8`
- `n_permutations = 3`

Then repeated `generate(random_search=False)` calls returned the remaining sampled combinations in order from the internal sampled pool, not from the full original grid.

`ParamSpace` uses an instance-local random generator. Pass `seed=` for reproducible legacy sampling; it does not read or mutate Python's module-global random state. This legacy helper is outside the artifact-backed `round_params` reproducibility contract.

Use `ParamSpace` only on the legacy UEL path. The advanced path uses [Advanced Search](Advanced-Search.md) primitives instead.

## `data_dict_to_numpy`

`data_dict_to_numpy()` converts the standard split keys from polars or pandas into numpy arrays.

```python
from limen.utils import data_dict_to_numpy

arrays = data_dict_to_numpy(data_dict)
```

On a live local manifest-prepared `data_dict` in this repo, it converted:

- `x_train` to shape `(3610, 24)`
- `y_train` to shape `(3610,)`
- `x_val` to shape `(428, 24)`
- `x_test` to shape `(884, 24)`

This helper supports sklearn-style or numpy-first model code.

`split_data_to_prep_output()` is the companion converter that builds `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, and `y_test` from split frames without mutating the passed split list or column list.

## `adf_test`

`adf_test()` runs an Augmented Dickey-Fuller stationarity test and returns an `AdfResult`.

```python
from limen.utils import adf_test

result = adf_test(series)
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

In a live synthetic-model run in this repo with `target_confidence=0.8`, it returned:

- coverage value `0.867`
- a threshold near zero on that particular synthetic setup
- a results frame with columns:
  - `datetime`
  - `prediction`
  - `uncertainty`
  - `is_confident`
  - `confidence_threshold`
  - `actual_value`
  - `confidence_score`

Use this as an optional downstream helper, not as part of the core UEL contract.

## `split_by_dates`

`split_by_dates()` partitions a datetime-indexed `polars.DataFrame` into train, val, and test by half-open `[start, end)` datetime windows. It is the helper that `Manifest.set_split_dates` calls when `split_dates` is configured; it is also exported for direct use when the splitter is needed outside the manifest pipeline.

```python
from datetime import datetime
from limen.data.utils import split_by_dates

train, val, test = split_by_dates(
    df,
    datetime(2024, 1, 1), datetime(2024, 7, 1),
    datetime(2024, 7, 1), datetime(2024, 10, 1),
    datetime(2024, 10, 1), datetime(2025, 1, 1),
)
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

- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for the legacy path that still uses `ParamSpace`.
- Continue to [Advanced Search](Advanced-Search.md) for the newer search abstractions that replace `ParamSpace` in artifact-backed runs.
- Continue to [Reference Architecture](Reference-Architecture.md) for `data_dict_to_numpy()` usage inside model code.
