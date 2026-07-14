# Fractional Differentiation

Fractional differentiation makes a price series stationary while keeping as much of its memory as possible. Full first differencing (returns) is stationary but erases the level information a model can learn from; the raw price keeps all its memory but is non-stationary. Fractional differentiation sits between the two: it applies a real-valued differencing order `d` between `0` and `1` and keeps the smallest amount of differencing that passes a stationarity test.

This guide covers when to reach for it, how to choose `d`, and how to wire it into an experiment. It follows Lopez de Prado's *Advances in Financial Machine Learning* (AFML), Chapter 5.

## What This Guide Covers

- why stationarity matters for a model and what memory you lose with plain differencing
- how to find the minimum differencing order with an ADF test
- how to add a fractionally differentiated feature to a manifest or a YAML experiment
- the exact output columns produced and the edge cases to know

## Prerequisites

- A working manifest or YAML experiment. Start with [Experiment Manifest](Experiment-Manifest.md) if you do not have one.
- The `stats` extra installed for the ADF helper: `pip install vaquum-limen[stats]`. `find_min_d` and `adf_test` depend on `statsmodels`.

## Why Stationarity, And Why Not Just Take Returns

A stationary series has a stable mean and variance over time. Most models assume this: a coefficient learned on 2021 price levels is meaningless on 2024 price levels, so a raw `close` column leaks regime information and generalizes badly.

The usual fix is to difference the series into returns (`d = 1`). That is stationary, but it is also memoryless — each row now describes only a one-step change and knows nothing about where the level sits in a longer move. Fractional differentiation keeps a fraction of that memory: at `d = 0.4`, each output value is a weighted sum of many past observations, with weights that decay but never vanish. You keep long-range structure and still clear a stationarity test.

The rule of thumb from AFML: use the **smallest** `d` that makes the series stationary. Anything larger throws away memory you did not need to spend.

## Finding The Minimum `d`

`find_min_d` sweeps `d` from `d_start` to `d_end` in steps of `step`, fractionally differencing the column at each value and testing the result with an Augmented Dickey-Fuller (ADF) test. It returns the first `d` whose series is stationary.

```python
import polars as pl

from limen.features import find_min_d

df = pl.DataFrame({'close': load_close_prices()})

d = find_min_d(
    df,
    col='close',
    d_start=0.0,
    d_end=1.0,
    step=0.1,
    significance_level=0.05,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `col` | — | column to difference and test |
| `d_start` | `0.0` | smallest `d` to try |
| `d_end` | `1.0` | largest `d` to try; also the fallback return value |
| `step` | `0.1` | increment between tested `d` values |
| `significance_level` | `0.05` | ADF p-value threshold for calling a series stationary |
| `threshold` | `1e-5` | weight-truncation threshold passed through to the differencing |

The return value is a single `float`: the smallest `d` that passed. If no tested `d` is stationary, it returns `d_end`. Very short series cannot be tested and also return `d_end` — the ADF test needs enough observations to be meaningful.

The stationarity call itself comes from `adf_test`, which wraps `statsmodels`' `adfuller` (with `autolag='AIC'`) and reports a small result object:

```python-fragment
from limen.utils import adf_test

result = adf_test(df['close'], significance_level=0.05)
result.stationary       # bool: True when p_value <= significance_level
result.p_value          # float
result.test_statistic   # float
result.critical_values  # dict[str, float]
```

`adf_test` returns a non-stationary, inconclusive result (`stationary=False`, `p_value=1.0`) for empty or constant series rather than raising, so a sweep over a degenerate column degrades gracefully. See [Utilities](Utilities.md) for the full `adf_test` and `AdfResult` reference.

### Recommended `d` Values

`find_min_d` gives you the data-driven answer for a specific series and window. When you would rather sweep `d` as an experiment parameter and let the search compare outcomes, a practical grid for hourly Bitcoin close prices is:

```python-fragment
'frac_diff_d': [0.0, 0.1, 0.2, 0.3, 0.4]
```

`d = 0.0` is the identity (a copy of the raw column — a useful baseline for comparison), and most Bitcoin price series reach stationarity well below `d = 0.5`. AFML also recommends applying fractional differentiation to **log** prices rather than raw prices for the cleanest results.

## Adding It As A Feature

`fractional_diff` is a feature-library function. It adds one new column per input column and preserves the originals, so indicators that depend on the raw price still work.

```python-fragment
def fractional_diff(data: pl.DataFrame,
                    d: float = 0.0,
                    cols: list[str] | None = None,
                    threshold: float = 1e-5) -> pl.DataFrame:
```

| Parameter | Default | Meaning |
|---|---|---|
| `d` | `0.0` | differencing order; `0` copies the column, `1` is a standard first difference |
| `cols` | `None` | columns to differentiate — **required**, a non-empty list |
| `threshold` | `1e-5` | drops fixed-width weights below this magnitude (AFML fixed-width FFD) |

It raises `ValueError` if `cols` is empty, if `d` is negative, or if `threshold` is not positive.

### In A Manifest

Pass the function to `add_feature`. A string value for `d` is resolved from the round's parameters, so the differencing order can be swept like any other parameter.

```python
from limen.data import HistoricalData
from limen.experiment import Manifest, MLManifest
from limen.features import fractional_diff
from limen.scalers import LogRegScaler
from limen.sfd.reference_architecture import logreg_binary
from limen.targets import QuantileBinaryTarget

def params():
    return {'frac_diff_d': [0.0, 0.2, 0.4]}

def manifest() -> Manifest:
    return (
        MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'row_count_limit': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_feature(fractional_diff, d='frac_diff_d', cols=['close'])
        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'close_fracdiff', 'quantile': 0.40},
            transform_params={'shift': -1},
        )
        .set_scaler(LogRegScaler)
        .with_reference_architecture(logreg_binary)
    )
```

### In A YAML Experiment

Reference the function by its resolvable path under a `features:` entry. The `{frac_diff_d}` placeholder binds to the `params:` block.

```yaml
    features:
      - func: limen.features.fractional_diff
        params:
          d: "{frac_diff_d}"
          cols:
            - close
```

```yaml
  params:
    frac_diff_d: [0.0, 0.2, 0.4]
```

For a fixed order, pass the number directly (`d: 0.5`) and omit the parameter.

## Output Columns

For each column in `cols`, `fractional_diff` adds `{col}_fracdiff` — so `close` produces `close_fracdiff`. The suffix is fixed (`FRACDIFF_SUFFIX = '_fracdiff'`). The original column is always kept.

When `d = 0.0`, the new column is a `Float64` copy of the original. When `d = 1.0`, it equals the standard first difference (`np.diff` with a leading null). For `0 < d < 1` it is the fixed-width fractionally differenced series, with leading rows null until enough history exists to fill the weight window.

NOTE: Because the leading rows are null until the weight window fills, a short split can end up without a valid `_fracdiff` column while a longer split has one. During split alignment the manifest drops the extra column so the feature schema stays consistent across train, validation, and test. See [Experiment Manifest](Experiment-Manifest.md) for the alignment behavior.

## Expected Artifacts

After a run, the fractionally differenced column appears in the prepared feature matrix and, if you select it as a model input, in the per-round records written to `round_data.jsonl`. When `d` is swept as a parameter, each `frac_diff_d` value is one row in `results.csv`, so the effect of differencing order on the metrics is visible directly. See [Log](Log.md) for reading those results.

## Where To Look

- `limen/features/fractional_diff.py` — `fractional_diff`, `find_min_d`, and the fixed-width weight helpers
- `limen/utils/adf_test.py` — `adf_test`, `AdfResult`
- `limen.features.fractional_diff` — the resolvable path for YAML `func:` and manifest `add_feature`

## Read Next

- [Features](Features.md) for the full feature-library reference, including the stationarity helpers.
- [Utilities](Utilities.md) for the `adf_test` and `AdfResult` reference.
- [Experiment Manifest](Experiment-Manifest.md) for `add_feature`, parameter references, and split alignment.
- [Perturbation Strategies](Perturbation-Strategies.md) to search feature and model choices robustly once the feature is in place.
