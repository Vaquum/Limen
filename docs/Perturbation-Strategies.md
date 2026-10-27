# Perturbation Strategies

A perturbation changes a declared part of preparation so the research can
compare its effect. The [Python project](examples/python-research/README.md)
chooses its feature columns and fitted transforms directly. The
[Compiled project](examples/compiled-research/README.md) uses native groups,
seeded ablation and PCA. Both retain their prepared partitions and compare
baseline and compressed Ridge fits on recorded observations.

These two cases prove their preparation and fitting behavior. They are not a
large feature-importance study or independent evidence that the chosen surface
will generalize.

## What This Guide Covers

- the five parameter-controlled perturbation primitives and the round parameters that drive them
- when to use each one, and which ones matter for linear, tree, and TabPFN models
- feature ablation in depth: determinism, protected columns, and how the dropped set is recorded
- deriving feature importance from an ablation sweep

## Prerequisites

- A working manifest or YAML experiment. Start with [Experiment Manifest](Experiment-Manifest.md) if you do not have one.
- Familiarity with parameter references: a string value in a manifest call (like `period='roc_period'`) is resolved from the current round's parameters. See [Experiment Manifest](Experiment-Manifest.md).

## The Five Primitives

Data perturbations can select a trailing `lookback_window` or one `sample_regime_blocks` block through `pre_split_data_selector`. Both preserve original row order and values. The [utility contracts](Utilities.md#temporal-perturbation-selectors) describe their bounds and seeded selection. `HistoricalData.get_spot_klines` and `get_spot_dollar_klines` also accept `lookback_days` with an explicit `end_date_limit`; combining it with `start_date_limit` is rejected before acquisition.

| Primitive | What it varies | Round parameter(s) | Manifest method |
|---|---|---|---|
| Data (rows) | temporal population and sampling clock | `lookback_days`, regime/seed controls, bar thresholds | `set_pre_split_data_selector()` / `set_bar_formation()` |
| Feature (columns) | selected groups, toggles and Drop-N subsets | `feature_groups`, inclusion keys, `feature_drop_count`, `feature_drop_seed` | `add_indicator()` / `add_feature()` / `set_feature_ablation()` |
| View (X values) | scaling, fractional differentiation and fitted representation | `scaler_type`, declared transform/shim parameters | `set_scaler_from_params()` / feature transforms and shims |
| Target (Y values) | target geometry, candidate jitter and training-label smoothing | horizon/threshold/barrier controls, `noise_seed`, `label_smoothing` | `with_target_class()` |
| Model (function) | learner complexity and optimization | architecture-specific hyperparameters | reference architecture and its consumed round parameters |

All five use the existing parameter grid and manifest references. `with_params_override()` remains a build-time tool for derived manifests; use a pre-split selector when the population must vary inside a sweep. Adaptive [imbalance/run bars](Data-Bars.md#adaptive-trade-point-bars) require individual observed trades. [Target perturbations](Targets.md#candidate-target-perturbations) retain their candidate identity through workers and reconstruction.

### Feature-Group Selection

Tag indicators and features with a `group=` name, then activate groups per round through the `feature_groups` parameter.

<!-- reference: perturbation-indicator source=limen/experiment/manifest_core.py region=signature:Manifest.add_indicator role=signature -->
```python
def add_indicator(self, func: Callable[..., Any], group: str | None=None, include_if: str | None=None, **params: Any) -> 'Manifest':
    ...
```

The `feature_groups` value is interpreted as follows: absent or `'all'` runs every group; a pipe-delimited string like `'momentum|volume'` runs only those groups. The value must be a string or a `TypeError` is raised. Ungrouped transforms always run regardless of the setting.

| Value | Active groups |
|---|---|
| absent or `all` | Every declared group. |
| `momentum` | Momentum only, plus ungrouped transforms. |
| `momentum&#124;price` | Momentum and price, plus ungrouped transforms. |

Use this to compare whole feature families — is momentum pulling its weight against volatility and volume, or is a single family carrying the signal?

### Conditional Toggle (`include_if`)

Gate a single feature on a boolean parameter.

<!-- reference: perturbation-feature source=limen/experiment/manifest_core.py region=signature:Manifest.add_feature role=signature -->
```python
def add_feature(self, func: Callable[..., Any], group: str | None=None, include_if: str | None=None, **params: Any) -> 'Manifest':
    ...
```

Declare boolean candidates for the `include_if` parameter; native preparation rejects other types.

The parameter must be a real boolean. A non-bool value raises `TypeError`, and — importantly — a **missing** key is treated as off, so the feature is excluded when the key is absent. Use `include_if` for a clean A/B on one feature, especially an expensive one you only want to pay for when it earns its place.

### Feature Ablation

Randomly drop `N` generated feature columns per round to probe robustness and importance.

<!-- reference: perturbation-ablation source=limen/experiment/manifest_core.py region=signature:MLManifest.set_feature_ablation role=signature -->
```python
def set_feature_ablation(self, drop_count_key: str='feature_drop_count', seed_key: str='feature_drop_seed') -> 'MLManifest':
    ...
```

| Parameter | Project case |
|---|---|
| `feature_drop_count` | `0` in the baseline, `1` in the compressed case. |
| `feature_drop_seed` | `7` in both cases. |

`set_feature_ablation()` accepts `drop_count_key` (default `'feature_drop_count'`) and `seed_key` (default `'feature_drop_seed'`) if you want different parameter names. The selection is deterministic for a given seed. Full behavior is in [Feature Ablation In Depth](#feature-ablation-in-depth) below.

### Scaler From Parameters

Choose the scaler per round instead of pinning one class.

<!-- reference: perturbation-scaler source=limen/experiment/manifest_core.py region=signature:MLManifest.set_scaler_from_params role=signature -->
```python
def set_scaler_from_params(self, param_name: str='scaler_type', extra_params: dict[str, Any] | None=None) -> 'MLManifest':
    ...
```

<!-- reference: perturbation-scaler-registry source=limen/scalers/registry.py region=definition:SCALER_REGISTRY role=definition -->
```python
SCALER_REGISTRY: dict[str, type] = {'linear': LinearScaler, 'logreg': LogRegScaler, 'robust': RobustScaler, 'rank_gauss': RankGaussScaler, 'causal_rolling_robust': CausalRollingRobustScaler}
```

The value is looked up in the scaler registry. The available keys are `linear`, `logreg`, `robust`, `rank_gauss`, and `causal_rolling_robust`. An unknown key raises `ValueError` listing the valid options. See [Scalers](Scalers.md) for what each one does. Use this to let the search pick the preprocessing that suits the model rather than assuming one is best.

### Manifest Override

`with_params_override(**overrides)` returns a deep copy of the manifest with selected values changed. It is a build-time variation, not a per-round search parameter.

<!-- reference: perturbation-override source=limen/experiment/manifest_core.py region=signature:Manifest.with_params_override role=signature -->
```python
def with_params_override(self, **overrides: Any) -> 'Manifest':
    ...
```

`split_config` is special-cased: it must be a 3-tuple of non-negative ints that are not all zero, and setting it clears any pinned `split_dates`. Every other key is validated against the data-source method's signature, so an unknown key raises `ValueError`. Use overrides to retrain a selected configuration on the full dataset, to run a fast smoke test on fewer rows, or to swap the data window.

NOTE: A sixth param-controlled transform, `set_pca_compression()`, gates PCA on a parameter (`auto_pca`) and sizes it with another (`pca_k`). It follows the same pattern as the primitives above; see [Experiment Manifest](Experiment-Manifest.md) for its parameters.

## Follow the preparation cases

### Python Research

The caller selects the drop explicitly and fits RobustScaler and optional two-component PCA only on its own training partition. The complete function also retains its label and stationarity diagnostics:

<!-- example: perturbation-python-preparation source=docs/examples/python-research/scenarios.py region=preparation scenario=search interface=python role=executed -->
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

The native Manifest chooses its generated-column drop using seed 7, applies the chosen set across partitions, then fits optional PCA on its finalized training surface. It retains native CCO behavior and does not claim boundary populations identical to the caller-owned case:

<!-- example: perturbation-compiled-preparation source=docs/examples/compiled-research/research.yaml region=preparation scenario=search interface=compiled role=executed -->
```yaml
    scaler:
      class: limen.scalers.RobustScaler
    feature_ablation:
      drop_count_key: feature_drop_count
      seed_key: feature_drop_seed
    pca_compression:
      enabled_param: auto_pca
      n_components_param: pca_k
```

| YAML location | Native choice |
|---|---|
| indicator/feature entries | `group` and optional `include_if` selection. |
| `sfd.manifest.scaler.from_params` | Search key selecting a registered scaler. |
| `sfd.manifest.feature_ablation` | Drop-count and seed key names. |
| `sfd.params` | Candidate lists for every selected search dimension. |

## Feature Ablation In Depth

Ablation drops randomly chosen feature columns so you can measure how much the model depends on any individual feature and whether it degrades gracefully.

**What is eligible.** Only columns *created by* your indicators and features are eligible to drop. The raw input columns of the training split — `datetime`, OHLCV, and anything present before transforms ran — are protected, and so is the target column. This guarantees ablation never removes the label or the data a downstream indicator still needs.

**Determinism.** For a given `(feature_drop_seed, feature_drop_count)` and a fixed set of eligible columns, the same columns are dropped every time. Selection sorts the eligible columns and samples with a seeded RNG, so results are reproducible across runs and machines.

**Cross-split consistency.** The drop set is chosen once on the training split and re-applied identically to validation and test, so every split sees the same feature matrix. When a promoted model runs inference, the recorded drop set is applied again — the sensor sees exactly the features the model was trained on.

**Edge cases.** `feature_drop_count` must be a non-negative int (booleans are rejected); `feature_drop_seed` must be an int. A drop count of `0` leaves all features intact; its result row records `[]`. Requesting more drops than there are eligible columns raises `ValueError`.

**What gets recorded.** When ablation is configured, every result row records `_dropped_features` as JSON text containing the sorted dropped names, or `[]` when none were dropped. CSV, the in-memory experiment log and parquet use that same encoding, including failed rounds. Successful `round_data.jsonl.gz` records retain the prepared list under `round_params._dropped_features`; zero-drop records omit that key. Decode the result field with `json.loads` for exact membership checks. Experiments without ablation keep their existing columns.

If preparation fails before ablation runs, `[]` means no drop names were recorded; it does not imply successful preparation. Read `strict_mode_error` alongside the field. Keep `_dropped_features` out of explicit `Log(..., cols_to_multilabel=...)` lists.

### Deriving Feature Importance From Ablation

Recorded drops permit a descriptive feature-sensitivity analysis. Association with a metric depends on the other searched choices and the observed population; it does not establish causal importance.

`_dropped_features` contains a JSON-encoded list in UEL result rows. Decode it and construct explicit per-feature boolean membership columns before calling the source-bound correlation API below. The two small preparation cases do not meet its default observation floor and publish no importance ranking.

<!-- reference: perturbation-correlation source=limen/log/_experiment_parameter_correlation.py region=signature:experiment_parameter_correlation role=signature -->
```python
def experiment_parameter_correlation(self: Any, metric: str, *, cols_to_drop: list[str] | None=None, sort_key: str | None=None, sort_ascending: bool=False, heads: Sequence[float | int] | None=None, method: Literal['pearson', 'kendall', 'spearman']='spearman', n_boot: int=300, min_n: int=10, random_state: int=0) -> pl.DataFrame:
    ...
```

`experiment_parameter_correlation` coerces the new boolean columns to `0`/`1` and reports their correlation with the metric. A negative correlation with a maximized metric describes lower recorded values when that feature was dropped. Near-zero association can also reflect insufficient variation, dependence or competing effects; it does not prove the feature unnecessary. See [Log](Log.md) for the full `experiment_parameter_correlation` reference, including its bootstrap confidence columns.

NOTE: This is an attribution signal, not a proof. Random feature ablation perturbs the pipeline and measures the metric response; it does not, by itself, establish that a feature carries out-of-sample predictive edge. Read it alongside walk-forward evidence and the model's own coefficients where available.

## Choosing By Model Type

The primitives are not equally useful for every model. A rough decision guide:

- **Linear models** (`logreg_binary`, `dlinear_regressor`) are scale-sensitive and collinearity-sensitive. Scaler-from-params matters most here — sweep `logreg`, `robust`, and `rank_gauss`. Feature-group selection and ablation help find and prune redundant or noisy inputs. Treat fractional differentiation as a separately evaluated feature choice (see [Fractional Differentiation](Fractional-Differentiation.md)).
- **Tree models** (`lightgbm_binary`, `xgboost_regressor`) often retain split order under monotone rescaling, but numerical methods and model settings still matter. Test the declared configuration rather than assuming every preprocessing choice is irrelevant. Feature-group selection and ablation are the primary robustness levers; `include_if` is useful for gating expensive optional features.
- **TabPFN** (`tabpfn_binary`) is sensitive to feature count and expects scaled inputs. The [optional CPU campaign](examples/compiled-research/README.md#optional-model-campaign) uses recorded observations, a pinned real checkpoint and an identified environment. Its one-feature example does not establish a general feature-count or compression advantage.

## Expected Artifacts

After an experiment with perturbations, each round in `results.csv` carries the parameter values that produced it — the active `feature_groups`, the `use_vwap` flag, the `scaler_type`, and (when ablation is configured) the JSON-encoded `_dropped_features` list — alongside the standard metrics. That lets you attribute metric differences back to the pipeline choice directly from the log. See [Log](Log.md) for reading and analyzing those results.

New-format ablation experiments resume with the same JSON column. A pre-fix CSV without `_dropped_features` cannot resume or accept appended ablation rounds: start a new experiment directory for checkpointed MSQ runs, or a fresh `<experiment_name>.csv` for standard and uncheckpointed MSQ runs. Older standard runs may contain Python list representations rather than JSON; use a fresh file for those too. This fix checks column presence, not historical cell formats. Historical drop sets are not inferred or rewritten.

## Where To Look

- `limen/experiment/manifest_core.py` — `add_indicator`, `add_feature`, `set_feature_ablation`, `set_scaler_from_params`, `with_params_override`, `set_pca_compression`
- `limen/scalers/registry.py` — the scaler registry keys used by `set_scaler_from_params`
- `limen/log/` — `experiment_parameter_correlation` for parameter-to-metric analysis

## Read Next

- [Experiment Manifest](Experiment-Manifest.md) for the full manifest builder reference and parameter-reference rules.
- [Scalers](Scalers.md) for what each registry scaler does.
- [Fractional Differentiation](Fractional-Differentiation.md) for stationarity-aware features.
- [Log](Log.md) for `experiment_parameter_correlation` and reading perturbation impact.
- [Advanced Search](Advanced-Search.md) and [Reducers And Feedback](Reducers-And-Feedback.md) for steering the search over these parameters.
