# Triple-Barrier Method

`TripleBarrierTarget` labels a bar by the first crossing in its future **close
prices**: upper barrier, lower barrier, or a completed vertical horizon with no
crossing. It does not inspect intrabar high/low ordering, charge fill costs or
execute a position. The label and the [trading policy](Backtest.md) answer
separate questions.

The [Python project](examples/python-research/README.md) applies the target
inside caller-owned partitions. The [Compiled project](examples/compiled-research/README.md)
uses it in native split preparation. Both compare a full feature surface with
an ablated, train-fitted PCA surface and fit Ridge to the ternary response. Its
continuous forecast estimates that label's conditional mean; it is neither a
calibrated profit probability nor a return in price units.

## What This Guide Covers

- why path-dependent labeling differs from a fixed-horizon return label
- how the profit, stop, and time barriers are defined and computed
- how the profit-to-stop ratio is set through the two barrier multiples
- attaching the target to a manifest and a YAML experiment

## Prerequisites

- A working manifest or YAML experiment. Start with [Experiment Manifest](Experiment-Manifest.md) if you do not have one.
- The target reads a close-price column (`close` by default), so the data source must provide it.

## Current Scope

The shipping `TripleBarrierTarget` produces a single **ternary** label: `1` when the upper (profit) barrier is touched first, `-1` when the lower (stop) barrier is touched first, and `0` when neither is touched before the time barrier. There is one labeling mode; the target does not emit meta-labeling or companion columns. Rows are `null` without a finite positive volatility estimate. At the end of a partition, an observed earlier crossing still produces its label; a row with neither a crossing nor its complete vertical horizon is `null`. Native preparation drops unavailable labels.

## Why Path-Dependent Labeling

A fixed-horizon label asks "what is the return exactly `H` bars from now?" That ignores the path: a position that ran to a profit target on bar 3 and a position that bled to a stop on bar 3 can show the same sign at bar `H`. The triple-barrier label instead asks "which outcome came first?" — profit, stop, or timeout — which is the question a real position answers. Training on that label teaches the model to separate setups that resolve favorably from those that get stopped out, rather than setups that merely drift the right way by an arbitrary deadline.

## The Three Barriers

**Upper and lower barriers are volatility-scaled.** At each bar, Limen estimates volatility as the EWMA standard deviation of close-to-close returns (controlled by `span` and `min_periods`) and places the barriers at a multiple of that estimate:

- upper (profit) barrier at `+upper_multiple × volatility`
- lower (stop) barrier at `-lower_multiple × volatility`

The width is fixed using the **entry bar's** volatility, so barriers adapt to the local regime — wider in turbulent stretches, tighter in calm ones. There is no fixed-percentage mode; sizing is always in volatility units.

**The vertical barrier is a bar count.** `max_horizon` sets how many bars forward the scan runs. The method walks forward from the entry bar; the first bar whose return crosses a horizontal barrier sets the label, and if `max_horizon` bars pass with no touch, the label is `0`.

### Setting The Reward-To-Risk Ratio

There is no separate risk-reward parameter — the profit-to-stop ratio *is* `upper_multiple : lower_multiple`. The default `1.0 / 1.0` is symmetric. To label for a 2:1 reward-to-risk setup, widen the profit side or tighten the stop:

<!-- reference: triple-target-constructor source=limen/targets/triple_barrier.py region=signature:TripleBarrierTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, upper_multiple: float=1.0, lower_multiple: float=1.0, max_horizon: int=10, span: int=100, min_periods: int=100, close_col: str='close') -> None:
    ...
```

The project fixes upper/lower multiples at `1.5 : 1`, horizon at three bars,
and volatility span/minimum at eight. Wider stops change first-crossing labels;
they can also turn a formerly stopped path into timeout. Multiples describe
label geometry and do not establish a strategy's realized reward/risk ratio.

## Attaching The Target

`TripleBarrierTarget` accepts a training frame for interface consistency but fits no distributional state in its constructor. It calculates volatility on the frame supplied to `transform`. The Python case creates the target once and transforms each partition separately:

<!-- example: triple-python-preparation source=docs/examples/python-research/scenarios.py region=preparation scenario=search interface=python role=executed -->
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

The source-bound transformation reference is:

<!-- reference: triple-target-transform source=limen/targets/triple_barrier.py region=signature:TripleBarrierTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, *, horizon_jitter: int=0, barrier_noise_std: float=0.0, noise_seed: int=0, permutation_id: str | None=None, training: bool=False) -> pl.DataFrame:
    ...
```

`train_data` and `target_name` are supplied by the manifest; the rest are the barrier settings above. Each of `upper_multiple`, `lower_multiple`, `max_horizon`, `span`, and `min_periods` must be positive or the constructor raises `ValueError`. See [Targets](Targets.md) for the full parameter table and null-row semantics.

### Compiled YAML target

The Compiled preparation variant declares its triple-barrier target in YAML, alongside the features, train-fitted scaling, ablation and optional PCA. Its boundary context differs from Python's independent partition preparation; matching constructor values do not imply matching rows.

<!-- example: triple-compiled-preparation source=docs/examples/compiled-research/preparation.yaml region=target scenario=search interface=compiled role=executed -->
```yaml
    target:
      name: barrier_label
      class: limen.targets.TripleBarrierTarget
      fit_params:
        upper_multiple: 1.5
        lower_multiple: 1.0
        max_horizon: 3
        span: 8
        min_periods: 8
```

| YAML field | Meaning |
|---|---|
| `sfd.manifest.target.name` | Name of the output label. |
| `target.class` | `limen.targets.TripleBarrierTarget` |
| `target.fit_params` | Constructor settings other than the supplied training frame and target name. |
| `sfd.params` | Candidate lists for any braced references in those settings. |

YAML uses the fields above. Numeric settings may be fixed or use braced references to declared candidate lists. Ternary labels require a compatible model contract; they cannot be substituted into an arbitrary binary-classification template.

## Choosing `span` And `min_periods`

`span` sets the EWMA window for the volatility estimate and `min_periods` sets the warmup before any volatility — and therefore any label — is emitted. Both consume rows: the first estimate needs `min_periods` close-to-close returns, and unavailable target rows are removed by native preparation. Admitted carried context can supply earlier observations for a later split. Size them well below the length of your smallest split, or a short validation or test split can be left with too few labeled rows. See the null-handling notes in [Targets](Targets.md).

## Expected Artifacts

The target adds one column, named by the `name` you pass, holding the ternary label (`Int8`, values in `{-1, 0, 1}` or `null`). It appears in prepared data. These project cases retain the actual partition matrices and labels as Parquet diagnostics; ordinary round recording does not promise a complete target matrix. Inspect class counts on training. Altering the horizon or multiples changes the research question and consumes selection evidence; a more balanced label is not itself a better target.

## Where To Look

- `limen/targets/triple_barrier.py` — the `TripleBarrierTarget` implementation and barrier scan
- `limen.targets.TripleBarrierTarget` — the resolvable path for YAML `class:` and manifest imports
- [Targets](Targets.md) — the full parameter table, null-row semantics, and the adjacent `RiskRewardRatioTarget`

## Read Next

- [Targets](Targets.md) for the complete target reference and other label constructions.
- [Experiment Manifest](Experiment-Manifest.md) for `with_target_label` and parameter references.
- [Perturbation Strategies](Perturbation-Strategies.md) to sweep barrier geometry and features robustly.
- [Backtest](Backtest.md) to evaluate how a labeled strategy would have traded.
