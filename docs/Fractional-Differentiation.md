# Fractional Differentiation

Fractional differentiation applies a backward-looking weighted difference to a
recorded series. The order and truncated history determine which level changes
remain in the feature. An ADF result describes one tested window; the transform
does not guarantee stationarity or predictive value.

The [Python project](examples/python-research/README.md) calls the feature
explicitly inside each partition and owns its fitted preprocessing. The
[Compiled project](examples/compiled-research/README.md) declares the same feature
inside a native Manifest, with split preparation, CCO, ablation and compression.
Their independent tests check the actual prepared values and training fits.

## What This Guide Covers

- why stationarity matters for a model and what memory you lose with plain differencing
- how to find the minimum differencing order with an ADF test
- how to add a fractionally differentiated feature to a manifest or a YAML experiment
- the exact output columns produced and the edge cases to know

## Prerequisites

- A working manifest or YAML experiment. Start with [Experiment Manifest](Experiment-Manifest.md) if you do not have one.
- The `stats` extra installed for the ADF helper: `pip install vaquum-limen[stats]`. `find_min_d` and `adf_test` depend on `statsmodels`.

## Why Stationarity, And Why Not Just Take Returns

Stationarity concerns a series' distribution over time. Stable mean and variance
are useful diagnostics, but do not establish strict stationarity. Differencing
price levels and taking returns are distinct transformations; neither is
universally stationary or memoryless.

At `d=0`, this helper copies the price column. At `d=1`, it subtracts the previous
price. Intermediate orders combine prior observations with recursively derived
weights. Fixed-width truncation discards sufficiently small weights, so the
implemented history is finite. Choosing a smaller order can retain more level
information; it does not prove that information is useful for the model.

## Finding The Minimum `d`

`find_min_d` sweeps `d` from `d_start` to `d_end` in steps of `step`, fractionally differencing the column at each value and testing the result with an Augmented Dickey-Fuller (ADF) test. It returns the first `d` whose transformed training window passes its ADF criterion.

<!-- reference: fractional-minimum-d source=limen/features/fractional_diff.py region=signature:find_min_d role=signature -->
```python
def find_min_d(data: pl.DataFrame, col: str, d_start: float=0.0, d_end: float=1.0, step: float=0.1, significance_level: float=0.05, threshold: float=1e-05) -> float:
    ...
```

| Parameter | Default | Meaning |
|---|---|---|
| `col` | — | column to difference and test |
| `d_start` | `0.0` | smallest `d` to try |
| `d_end` | `1.0` | largest `d` to try; also the fallback return value |
| `step` | `0.1` | increment between tested `d` values |
| `significance_level` | `0.05` | ADF p-value threshold for calling a series stationary |
| `threshold` | `1e-5` | weight-truncation threshold passed through to the differencing |

The return value is a single `float`: the first tested `d` that passed. If no tested `d` passes, it returns `d_end`. Very short series cannot be tested and also return `d_end` — the ADF test needs enough observations to be meaningful.

The stationarity call itself comes from `adf_test`, which wraps `statsmodels`' `adfuller` (with `autolag='AIC'`) and reports a small result object:

<!-- reference: fractional-adf source=limen/utils/adf_test.py region=signature:adf_test role=signature -->
```python
def adf_test(series: pl.Series, significance_level: float=0.05) -> AdfResult:
    ...
```

`adf_test` returns a non-stationary, inconclusive result (`stationary=False`, `p_value=1.0`) for empty or constant series rather than raising, so a sweep over a degenerate column degrades gracefully. See [Utilities](Utilities.md) for the full `adf_test` and `AdfResult` reference.

### Choose the order and window

An order chosen on one series and window is not a transferable stationarity
claim. Keep diagnostic selection inside training and retain the test result as
well as the returned order: `d_end` also represents the no-passing-order case.

Choose a finite order grid and weight threshold before inspecting assessment results. The Python project diagnoses the training-only grid `0.0`, `0.2`, `0.4`; both projects separately use fixed order `0.4` and truncation threshold `0.05` for their prepared-matrix comparison.

Applying the helper to log prices is a separate modeling choice. Declare that
transformation explicitly rather than treating the raw-close project case as a
reproduction of every AFML recommendation.

## Adding It As A Feature

`fractional_diff` is a feature-library function. It adds one new column per input column and preserves the originals, so indicators that depend on the raw price still work.

<!-- reference: fractional-feature source=limen/features/fractional_diff.py region=signature:fractional_diff role=signature -->
```python
def fractional_diff(data: pl.DataFrame, d: float=0.0, cols: list[str] | None=None, threshold: float=1e-05) -> pl.DataFrame:
    ...
```

| Parameter | Default | Meaning |
|---|---|---|
| `d` | `0.0` | differencing order; `0` copies the column, `1` is a standard first difference |
| `cols` | `None` | columns to differentiate — **required**, a non-empty list |
| `threshold` | `1e-5` | drops fixed-width weights below this magnitude (AFML fixed-width FFD) |

It raises `ValueError` if `cols` is empty, if `d` is negative, or if `threshold` is not positive.

### Python Research

This is the complete Python preparation case used by its search scenario. Its training-only ADF diagnostic and fixed-order comparison are separate decisions. Execute it through the complete project, including its imports and recorded-data helpers:

<!-- example: fractional-python-preparation source=docs/examples/python-research/scenarios.py region=preparation scenario=search interface=python role=executed -->
```python
def preparation(data_dir, output):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from limen.features import fractional_diff, find_min_d
    from limen.indicators import roc
    from limen.scalers import RobustScaler
    from limen.targets import TripleBarrierTarget
    from limen.utils import adf_test

    raw = hourly(data_dir, 1440)
    partitions = [raw[:864], raw[864:1152], raw[1152:1440]]
    d = find_min_d(partitions[0], 'close', d_end=.4, step=.2, threshold=.05)
    stationarity = adf_test(fractional_diff(partitions[0], d=d, cols=['close'], threshold=.05)['close_fracdiff'])
    cases = []
    for compressed in (False, True):
        name = 'compressed' if compressed else 'baseline'
        directory = output / name
        directory.mkdir(parents=True)
        target = TripleBarrierTarget(partitions[0], 'label', upper_multiple=1.5,
                                     lower_multiple=1., max_horizon=3, span=8, min_periods=8)
        columns = ['close', 'roc_3', 'roc_12', 'close_fracdiff']
        dropped = ['roc_12'] if compressed else []
        features = [column for column in columns if column not in dropped]
        prepared = []
        for partition in partitions:
            frame = fractional_diff(roc(roc(partition, period=3), period=12),
                                    d=.4, cols=['close'], threshold=.05)
            prepared.append(target.transform(frame).select('datetime', *features, 'label').drop_nulls())
        scaler = RobustScaler(prepared[0].select(features))
        scaled = [scaler.transform(frame.select(features)) for frame in prepared]
        pca = PCA(n_components=2, svd_solver='full') if compressed else None
        matrices = ([pca.fit_transform(scaled[0].to_numpy()),
                     *[pca.transform(frame.to_numpy()) for frame in scaled[1:]]]
                    if pca is not None else [frame.to_numpy() for frame in scaled])
        names = ['pc_0', 'pc_1'] if compressed else features
        model = Ridge(alpha=1., solver='svd').fit(matrices[0], prepared[0]['label'].to_numpy())
        predictions = model.predict(matrices[2])
        for split, frame, matrix in zip(('train', 'val', 'test'), prepared, matrices, strict=True):
            pl.DataFrame(matrix, schema=names, orient='row').with_columns(
                frame['datetime'], frame['label']).write_parquet(directory / (split + '.parquet'))
        cases.append({'name': name, 'features': features, 'dropped': dropped,
                      'rows': [frame.height for frame in prepared],
                      'medians': scaler.medians, 'iqrs': scaler.iqrs,
                      'pca_mean': None if pca is None else pca.mean_.tolist(),
                      'components': None if pca is None else pca.components_.tolist(),
                      'predictions': predictions.tolist(),
                      'rmse': float(np.sqrt(np.mean((predictions-prepared[2]['label'].to_numpy())**2)))})
    return {'cases': cases, 'partition_rows': [864, 288, 288], 'carried_context': False,
            'fixed_d': .4, 'weight_threshold': .05, 'adf_grid_result': d,
            'adf_stationary': bool(stationarity.stationary), 'adf_p_value': float(stationarity.p_value)}
```

### Compiled Research

The Compiled YAML declares fixed-order fractional differentiation as a feature. The pipeline carries its admitted prior context into later partitions and fits robust scaling and optional PCA only on training. The complete project executes this declaration through the CLI:

<!-- example: fractional-compiled-preparation source=docs/examples/compiled-research/research.yaml region=features scenario=search interface=compiled role=executed -->
```yaml
    features:
      - func: limen.features.lag_range
        params:
          col: roc_1
          start: 1
          end: 3
          group: momentum
      - func: limen.features.fractional_diff
        params:
          d: 0.4
          cols: [close]
          threshold: 0.05
          group: price
```

| YAML field | Binding |
|---|---|
| `sfd.manifest.features[].func` | `limen.features.fractional_diff` |
| `features[].params.d` | A fixed numeric order or a braced search-parameter reference. |
| `features[].params.cols` | The nonempty source-column list. |
| `features[].params.threshold` | A finite positive truncation threshold. |
| `sfd.params` | Declared candidate lists for any referenced values. |

YAML resolves the same feature through the fields above; a fixed numeric order requires no search reference.

## Output Columns

For each column in `cols`, `fractional_diff` adds `{col}_fracdiff` — so `close` produces `close_fracdiff`. The suffix is fixed (`FRACDIFF_SUFFIX = '_fracdiff'`). The original column is always kept.

When `d = 0.0`, the new column is a `Float64` copy of the original. When `d = 1.0`, it equals the standard first difference (`np.diff` with a leading null). For `0 < d < 1` it is the fixed-width fractionally differenced series, with leading rows null until enough history exists to fill the weight window.

NOTE: Because the leading rows are null until the weight window fills, a short split can end up without a valid `_fracdiff` column while a longer split has one. During split alignment the manifest drops the extra column so the feature schema stays consistent across train, validation, and test. See [Experiment Manifest](Experiment-Manifest.md) for the alignment behavior.

## Expected Artifacts

The project cases save `baseline/` and `compressed/` prepared Parquet partitions under their search scenario's `preparation/` directory. The baseline retains the fractional feature; the compressed case drops `roc_12` and replaces the surviving matrix with two fitted components. Ordinary round logs do not promise to serialize prepared feature matrices. A UEL sweep records its order parameter and metrics in the ordinary results; its payloads follow the [recording policy](Universal-Experiment-Loop.md#recording-policy).

## Where To Look

- `limen/features/fractional_diff.py` — `fractional_diff`, `find_min_d`, and the fixed-width weight helpers
- `limen/utils/adf_test.py` — `adf_test`, `AdfResult`
- `limen.features.fractional_diff` — the resolvable path for YAML `func:` and manifest `add_feature`

## Read Next

- [Features](Features.md) for the full feature-library reference, including the stationarity helpers.
- [Utilities](Utilities.md) for the `adf_test` and `AdfResult` reference.
- [Experiment Manifest](Experiment-Manifest.md) for `add_feature`, parameter references, and split alignment.
- [Perturbation Strategies](Perturbation-Strategies.md) to search feature and model choices robustly once the feature is in place.
