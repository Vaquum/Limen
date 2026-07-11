# Perturbation Strategies

A perturbation is a parameter-controlled variation of the feature and preprocessing pipeline, searched *within* a single experiment. Instead of hand-writing a new manifest for every "what if we dropped this feature" or "what if we scaled differently" question, you declare the variation once, expose it as a parameter, and let the search compare outcomes across rounds. Perturbations are how Limen tests robustness, attributes value to feature families, and avoids over-fitting to one fixed pipeline.

This guide covers the five perturbation primitives, when to reach for each, how they differ by model type, and how to read feature importance out of an ablation sweep.

## What This Guide Covers

- the five parameter-controlled perturbation primitives and the round parameters that drive them
- when to use each one, and which ones matter for linear, tree, and TabPFN models
- feature ablation in depth: determinism, protected columns, and how the dropped set is recorded
- deriving feature importance from an ablation sweep

## Prerequisites

- A working manifest or YAML experiment. Start with [Experiment Manifest](Experiment-Manifest.md) if you do not have one.
- Familiarity with parameter references: a string value in a manifest call (like `period='roc_period'`) is resolved from the current round's parameters. See [Experiment Manifest](Experiment-Manifest.md).

## The Five Primitives

| Primitive | What it varies | Round parameter(s) | Manifest method |
|---|---|---|---|
| Feature-group selection | which declared feature families run | `feature_groups` | `add_indicator(..., group=)` / `add_feature(..., group=)` |
| Conditional toggle | whether one optional feature runs | the `include_if` key | `add_feature(..., include_if=)` |
| Feature ablation | randomly drops N generated feature columns | `feature_drop_count`, `feature_drop_seed` | `set_feature_ablation()` |
| Scaler from parameters | which scaler preprocesses the matrix | `scaler_type` | `set_scaler_from_params()` |
| Manifest override | split ratios or data-source params | — (build-time) | `with_params_override()` |

The first four are searched across rounds through the `params()` grid. The fifth builds a derived manifest and is used for retrains, smoke runs, and alternate windows rather than in-experiment search.

### Feature-Group Selection

Tag indicators and features with a `group=` name, then activate groups per round through the `feature_groups` parameter.

```python-fragment
.add_indicator(roc, group='momentum', period=14)
.add_indicator(atr, group='volatility', period=14)
.add_feature(vwap, group='volume')
```

The `feature_groups` value is interpreted as follows: absent or `'all'` runs every group; a pipe-delimited string like `'momentum|volume'` runs only those groups. The value must be a string or a `TypeError` is raised. Ungrouped transforms always run regardless of the setting.

```python-fragment
'feature_groups': ['all', 'momentum', 'momentum|volume']
```

Use this to compare whole feature families — is momentum pulling its weight against volatility and volume, or is a single family carrying the signal?

### Conditional Toggle (`include_if`)

Gate a single feature on a boolean parameter.

```python-fragment
.add_feature(vwap, group='volume', include_if='use_vwap')
```

```python-fragment
'use_vwap': [True, False]
```

The parameter must be a real boolean. A non-bool value raises `TypeError`, and — importantly — a **missing** key is treated as off, so the feature is excluded when the key is absent. Use `include_if` for a clean A/B on one feature, especially an expensive one you only want to pay for when it earns its place.

### Feature Ablation

Randomly drop `N` generated feature columns per round to probe robustness and importance.

```python-fragment
.set_feature_ablation()
```

```python-fragment
'feature_drop_count': [0, 1, 2]
'feature_drop_seed':  [42]
```

`set_feature_ablation()` accepts `drop_count_key` (default `'feature_drop_count'`) and `seed_key` (default `'feature_drop_seed'`) if you want different parameter names. The selection is deterministic for a given seed. Full behavior is in [Feature Ablation In Depth](#feature-ablation-in-depth) below.

### Scaler From Parameters

Choose the scaler per round instead of pinning one class.

```python-fragment
.set_scaler_from_params('scaler_type')
```

```python-fragment
'scaler_type': ['logreg', 'robust', 'rank_gauss']
```

The value is looked up in the scaler registry. The available keys are `linear`, `logreg`, `robust`, `rank_gauss`, and `causal_rolling_robust`. An unknown key raises `ValueError` listing the valid options. See [Scalers](Scalers.md) for what each one does. Use this to let the search pick the preprocessing that suits the model rather than assuming one is best.

### Manifest Override

`with_params_override(**overrides)` returns a deep copy of the manifest with selected values changed. It is a build-time variation, not a per-round search parameter.

```python-fragment
full_data = manifest.with_params_override(split_config=(1, 0, 0))
smoke     = manifest.with_params_override(row_count_limit=5000)
```

`split_config` is special-cased: it must be a 3-tuple of non-negative ints that are not all zero, and setting it clears any pinned `split_dates`. Every other key is validated against the data-source method's signature, so an unknown key raises `ValueError`. Use overrides to retrain a selected configuration on the full dataset, to run a fast smoke test on fewer rows, or to swap the data window.

NOTE: A sixth param-controlled transform, `set_pca_compression()`, gates PCA on a parameter (`auto_pca`) and sizes it with another (`pca_k`). It follows the same pattern as the primitives above; see [Experiment Manifest](Experiment-Manifest.md) for its parameters.

## A Combined Example

The four in-experiment primitives compose in one manifest. Each is driven by a parameter, so a single experiment sweeps feature families, an optional feature, the scaler, and an ablation level together.

```python
from limen.data import HistoricalData
from limen.experiment import Manifest, MLManifest
from limen.features import vwap
from limen.indicators import atr, roc
from limen.sfd.reference_architecture import logreg_binary
from limen.targets import QuantileBinaryTarget


def params():
    return {
        'feature_groups':     ['all', 'momentum'],
        'use_vwap':           [True, False],
        'scaler_type':        ['robust', 'rank_gauss'],
        'feature_drop_count': [0, 1],
        'feature_drop_seed':  [42],
    }


def manifest() -> Manifest:
    return (
        MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'row_count_limit': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(roc, group='momentum', period=14)
        .add_indicator(atr, group='volatility', period=14)
        .add_feature(vwap, group='volume', include_if='use_vwap')
        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'roc_14', 'quantile': 0.40},
            transform_params={'shift': -1},
        )
        .set_scaler_from_params('scaler_type')
        .set_feature_ablation()
        .with_reference_architecture(logreg_binary)
    )
```

### As A YAML Experiment

The same perturbations are expressed declaratively. Groups and `include_if` live on the feature/indicator entries, ablation and the scaler are their own manifest blocks, and every driver is a list under `params`.

```yaml
    indicators:
      - func: limen.indicators.roc
        params:
          period: 14
          group: momentum
      - func: limen.indicators.atr
        params:
          period: 14
          group: volatility
    features:
      - func: limen.features.vwap
        include_if: use_vwap
        params:
          group: volume
    scaler:
      from_params: scaler_type
    feature_ablation:
      drop_count_key: feature_drop_count
      seed_key: feature_drop_seed
```

```yaml
  params:
    feature_groups: [all, momentum]
    use_vwap: [true, false]
    scaler_type: [robust, rank_gauss]
    feature_drop_count: [0, 1]
    feature_drop_seed: [42]
```

## Feature Ablation In Depth

Ablation drops randomly chosen feature columns so you can measure how much the model depends on any individual feature and whether it degrades gracefully.

**What is eligible.** Only columns *created by* your indicators and features are eligible to drop. The raw input columns of the training split — `datetime`, OHLCV, and anything present before transforms ran — are protected, and so is the target column. This guarantees ablation never removes the label or the data a downstream indicator still needs.

**Determinism.** For a given `(feature_drop_seed, feature_drop_count)` and a fixed set of eligible columns, the same columns are dropped every time. Selection sorts the eligible columns and samples with a seeded RNG, so results are reproducible across runs and machines.

**Cross-split consistency.** The drop set is chosen once on the training split and re-applied identically to validation and test, so every split sees the same feature matrix. When a promoted model runs inference, the recorded drop set is applied again — the sensor sees exactly the features the model was trained on.

**Edge cases.** `feature_drop_count` must be a non-negative int (booleans are rejected); `feature_drop_seed` must be an int. A drop count of `0` is a no-op and records nothing. Requesting more drops than there are eligible columns raises `ValueError`.

**What gets recorded.** On each round with a non-zero drop count, the sorted list of dropped column names is written into the round parameters under `_dropped_features`. That key flows into `results.csv` as a column and into `round_data.jsonl` as part of the round record, so every round carries an exact account of which features it ran without.

### Deriving Feature Importance From Ablation

Because the dropped set is recorded per round, an ablation sweep doubles as a feature-importance probe: if dropping a feature reliably hurts a metric, that feature matters.

`_dropped_features` is stored as a *list*, so it cannot be fed directly to the numeric correlation helper — build a per-feature boolean membership column first, then correlate it with the metric.

```python-fragment
# `log` is a Log with an experiment_log DataFrame from a completed ablation sweep.
candidates = ['roc_14', 'atr_14', 'vwap']

for feature in candidates:
    log.experiment_log[f"dropped_{feature}"] = (
        log.experiment_log['_dropped_features']
        .apply(lambda dropped: feature in (dropped or []))
    )

importance = log.experiment_parameter_correlation(metric='auc')
```

`experiment_parameter_correlation` coerces the new boolean columns to `0`/`1` and reports their correlation with the metric. A **negative** correlation between `dropped_<feature>` and the metric means dropping that feature tends to lower the metric — evidence the feature is important. A correlation near zero means the model does not miss it. See [Log](Log.md) for the full `experiment_parameter_correlation` reference, including its bootstrap confidence columns.

NOTE: This is an attribution signal, not a proof. Random feature ablation perturbs the pipeline and measures the metric response; it does not, by itself, establish that a feature carries out-of-sample predictive edge. Read it alongside walk-forward evidence and the model's own coefficients where available.

## Choosing By Model Type

The primitives are not equally useful for every model. A rough decision guide:

- **Linear models** (`logreg_binary`, `dlinear_regressor`) are scale-sensitive and collinearity-sensitive. Scaler-from-params matters most here — sweep `logreg`, `robust`, and `rank_gauss`. Feature-group selection and ablation help find and prune redundant or noisy inputs. Pair with fractional differentiation for stationarity (see [Fractional Differentiation](Fractional-Differentiation.md)).
- **Tree models** (`lightgbm_binary`, `xgboost_regressor`) are scale-invariant, so scaler-from-params usually adds nothing — leave the scaler out or fixed. Feature-group selection and ablation are the primary robustness levers; `include_if` is useful for gating expensive optional features.
- **TabPFN** (`tabpfn_binary`) is sensitive to feature count and expects scaled inputs. Keep the input set modest, pin an explicit scaler such as `robust`, and use ablation to confirm the model is not leaning on one fragile column.

## Expected Artefacts

After an experiment with perturbations, each round in `results.csv` carries the parameter values that produced it — the active `feature_groups`, the `use_vwap` flag, the `scaler_type`, and (for ablation rounds) the `_dropped_features` list — alongside the standard metrics. That lets you attribute metric differences back to the pipeline choice directly from the log. See [Log](Log.md) for reading and analyzing those results.

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
