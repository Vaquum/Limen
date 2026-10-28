# Reference Architecture

Reference architectures define the native model contract used by Compiled Research and Trainer. The [Compiled project](examples/compiled-research/README.md) exercises logistic selection, native Ridge preparation and the recorded model-family cases. The [Python project](examples/python-research/README.md) instead owns its estimator and score; ordinary Python models need not subclass `ReferenceModel`.

This page covers:

- what a `ReferenceModel` must implement
- how the built-in model classes behave
- what `evaluate(data, inline_metrics=True)` adds
- how class-based models relate to the simpler function wrappers used in manifests

## Public Surface

The current public reference-architecture exports are:

- `ReferenceModel`
- `EntropyVaRRegressor`
- `DLinearRegressor`
- `DirectionSizingModel`
- `LSTMBinary`
- `GaussianHMMBinary`
- `LightGBMBinary`
- `LogRegBinary`
- `RandomBinary`
- `RBFSVMBinary`
- `RidgeRegressor`
- `RuleBasedStrategy`
- `XGBoostRegressor`
- `TabPFNBinary`

Each model module also exposes a function-style wrapper with the same behavioral surface used by foundational manifests. The TabPFN symbols are importable without the optional dependency; constructing or training `TabPFNBinary` requires `vaquum-limen[tabpfn]`. The separate [TabPFN campaign](examples/compiled-research/README.md#optional-model-campaign) records a real pinned-checkpoint CPU run, its environment and numerical assertions. It does not claim CUDA execution or deterministic reconstruction.

The function exports are `direction_sizing`, `dlinear_regressor`, `entropy_var_regressor`, `gaussian_hmm_binary`, `lightgbm_binary`, `logreg_binary`, `lstm_binary`, `random_binary`, `rbf_svm_binary`, `ridge_regressor`, `rule_based`, `tabpfn_binary`, and `xgboost_regressor`.

## Caller-owned fitting

The Python project fits and retains its own scaler/estimator pipeline. Its
selection score, fitted-state identity and Sensor output meaning remain the
caller's decisions:

<!-- example: architecture-python-fit source=docs/examples/python-research/research.py region=fitted_scaler scenario=selection interface=python role=executed -->
```python
def fit(train, round_params):
    return make_pipeline(StandardScaler(), Ridge(alpha=round_params["alpha"])).fit(
        train.select("momentum"), train["next_return"].to_numpy()
    )
```

## `ReferenceModel`

`ReferenceModel` is the base contract. Every subclass must implement:

<!-- reference: architecture-base-train source=limen/sfd/reference_architecture/base.py region=signature:ReferenceModel.train role=signature -->
```python
@abstractmethod
def train(self, data: dict[str, Any], **params: Any) -> 'ReferenceModel':
    ...
```

<!-- reference: architecture-base-predict source=limen/sfd/reference_architecture/base.py region=signature:ReferenceModel.predict role=signature -->
```python
@abstractmethod
def predict(self, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

<!-- reference: architecture-base-evaluate source=limen/sfd/reference_architecture/base.py region=signature:ReferenceModel.evaluate role=signature -->
```python
@abstractmethod
def evaluate(self, data: dict[str, Any], inline_metrics: bool=True) -> dict[str, Any]:
    ...
```

### Data expectations

Manifest preparation can provide:

- `x_train`, `y_train`
- `x_val`, `y_val`
- `x_test`, `y_test`
- `price_data_for_backtest`
- `_feature_names`
- `_alignment`
- `_scaler`

Models consume the subset of keys they need from the standard Limen shape.

### Sklearn input boundary

Manifest preparation retains Polars frames and series. Uncalibrated and
threshold-only `LogRegBinary` and `RidgeRegressor` convert features and labels to
NumPy arrays at sklearn fit and prediction calls, preserving row and column order.
Conversion does not replace prepared entries or recording feature names.

When a probability-calibration callable is configured, LogReg training and
calibration fitting/prediction retain their original input containers. Native
array inputs remain arrays. Sklearn's calibration fold gathers can change
floating-point prediction arithmetic when frames are converted first; isotonic
calibration can turn that difference into a different selected threshold.
Frame and array calibration decisions are therefore not promised identical.

Custom calibration callbacks and fitted predictors receive original feature
inputs; threshold callbacks retain original validation labels. Calibration keeps
its existing dataframe-protocol dependency and backend warnings, including
feature-name warnings when a frame-trained model later receives arrays in Sensor.
Backend errors and warnings continue to propagate. This boundary applies to the
same model calls in serial and worker execution; it adds no execution mode or
preparation reuse. The ordinary per-permutation path remains the default, and
factorized snapshot admission follows its explicit [reuse contract](Universal-Experiment-Loop.md#reuse-declarations).

## The Built-In Model Classes

| Class | Task shape | Deterministic | Notes |
|---|---|---|---|
| `LogRegBinary` | binary classification | no | sklearn logistic regression wrapper; manifest wrapper exposes constructor params; solver refits are not bit-reproducible across BLAS builds and thread counts, so Trainer validates with relative tolerance |
| `LightGBMBinary` | binary classification | yes | LightGBM classifier exposing the full `LGBMClassifier` surface; accepts `binary`, `cross_entropy`, or `None` as its objective and rejects other objectives before training; early stopping on the validation split; reproducibility pinned via `deterministic`/`force_row_wise`/`random_state` defaults |
| `RandomBinary` | binary baseline | no | intentionally stochastic |
| `RidgeRegressor` | regression | no | sklearn Ridge wrapper; conservative flag because stochastic solvers are supported |
| `XGBoostRegressor` | regression | yes | requires `xgboost`; training is bit-reproducible for a fixed `random_state` on a fixed environment (verified byte-identical predictions across processes and thread counts), so Trainer validates with its near-exact deterministic tolerance |
| `DLinearRegressor` | regression | yes | canonical DLinear semantics; closed-form SVD ridge fit, no seed; requires `scipy` |
| `TabPFNBinary` | binary classification | no | optional, requires `tabpfn` |
| `RuleBasedStrategy` | rule-based long/flat | yes | no training step; boolean predicate logic |

The `deterministic` flag matters because [Trainer](Trainer.md) uses it to choose its validation tolerance.

## `RidgeRegressor`

`RidgeRegressor` delegates fitting and prediction to `sklearn.linear_model.Ridge`.
It uses `x_train` and `y_train` for fitting; validation and test targets do not
participate in training. Each `train()` call constructs a fresh estimator.
The fitted sklearn model remains available as `model.model`.

The function wrapper exposes `alpha`, `fit_intercept`, `copy_X`, `max_iter`,
`tol`, `solver`, `positive`, and `random_state`. Defaults follow sklearn except
that the function wrapper seeds stochastic solvers with `random_state=42`.
`tol` retains sklearn's meaning: it is not a Cholesky pivot cutoff, and it has
no effect for `cholesky` or `svd`. Backend errors and warnings are not replaced
with alternate results. See the [sklearn Ridge reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
for solver-specific behavior.

`predict({'x_test': features})` returns continuous `_preds`. `evaluate()` uses
Limen's existing continuous metrics; with `inline_metrics=True`, it also uses
`preds > 0` and `y_test > 0` for directional confusion metrics and the standard
long/flat backtest when prices are supplied. Costs and execution remain owned
by the manifest/backtest layer. The function wrapper returns `_model` for
normal Trainer reconstruction. Scaling belongs to the manifest, not the model.

## Direction And Sizing

`DirectionSizingModel` owns two fitted components and emits one signed exposure in `[-1, 1]`. Its native wrapper is `direction_sizing`; default components are sklearn LogisticRegression for direction and Ridge for magnitude. It requires [private simulated-trade outcomes](Targets.md#simulated-trade-outcomes) and `prediction_mode='target_exposure'`.

<!-- example: architecture-direction-sizing source=docs/examples/compiled-research/family_direction_sizing.yaml region=__file__ scenario=search interface=compiled role=executed -->
```yaml
# region declaration
schema_version: '1.0'
metadata:
  name: book_direction_sizing
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_15m_20250101_20250531.parquet
    split_dates:
      train_start: '2025-01-01'
      train_end: '2025-01-15'
      val_start: '2025-01-15'
      val_end: '2025-01-18'
      test_start: '2025-01-18'
      test_end: '2025-01-22'
    indicators:
    - func: limen.indicators.roc
      params:
        period: 1
        group: momentum
    target:
      name: outcome
      class: limen.targets.TradeOutcomeTarget
    backtest:
      prediction_mode: target_exposure
      product:
        kind: linear_perpetual
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1e-09
        min_notional: 0
      source_interval_seconds: 900
      fee_bps: 1
      slip_bps: 0
      max_holding_seconds: 1800
    scaler:
      class: limen.scalers.RobustScaler
    reference_architecture: limen.sfd.reference_architecture.direction_sizing
  params:
    direction_params:
    - max_iter: 1000
      C: 0.5
    sizing_params:
    - alpha: 1.0
    conditional_size:
    - false
    - true
    folds:
    - 3
    min_train_samples:
    - 2
    seed:
    - 42
uel:
  output_path: compiled-family_direction_sizing
  n_permutations: 2
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
# endregion declaration
```

This is the Compiled project's actual direction/sizing definition. The CLI reads its literal bundled input path and executes both `conditional_size` candidates. The declared source interval is 900 seconds and the holding limit is 1,800 seconds. Inputs require recorded OHLC and UTC datetime with that regular interval, or explicit causal interval metadata. Funding is optional and independently [tunable](Backtest.md#signed-exposure-elapsed-exits-and-funding).

When both candidates complete, direction is the strictly better positive side. Equal returns and neither side profitable map to flat. Magnitude is the better return divided by positive `return_scale` (default `0.01`), clipped to `max_size` (default `1`). One unavailable side excludes that row from paired fitting/metrics. Predictions still cover every feature-valid inference row.

`direction_params` and `sizing_params` have separate search namespaces. `direction_features` and `sizing_features` select prepared feature names independently. Factories accept `seed` and component kwargs and return estimators with `fit(x, y)`/`predict(x)`. They require importable module-qualified identities and a `deterministic` declaration. Native manifests accept callables; YAML component factory paths use the existing native resolver. Unknown kwargs, invalid subsets, nonfinite predictions and insufficient classes/samples fail explicitly.

Set `conditional_size=True` to add direction to the sizing inputs. Training then uses chronological out-of-fold direction predictions. Each fold refits automatic feature shims and its scaler/PCA on its earlier prefix, then fits direction only on feature-valid outcomes completed before the fold begins. Automatic-shim warmup cannot change the original chronological fold boundaries; unavailable rows are excluded by source identity. The final outer fitted state remains the inference state. The initial fold has no causal predictions and cannot train sizing. Outer validation/test never calibrate this stack. `folds` (default `5`) and `min_train_samples` (default `20`) are tunable. Final direction uses all eligible training outcomes.

`direction_accuracy` and `size_mae` use completed paired test outcomes; inline execution metrics retain the entire inference population. Changed sizing signals target post-cost current equity. Unchanged sizing signals hold quantity, as defined by the shared execution rules.

JSONL binds the architecture, factory source hashes, component params, seed, feature/label sources, outcome mapping and economics to the round and manifest. Trainer reconstructs both components and rejects changed identities before tolerant metric checks. Sensor preserves the exact signed scalar, actual availability and read-only frozen contract. Binary Cohort rejects these members, including a single member. Use Sensor directly; downstream replay/paper/live parity remains a separate integration request.

## Probability Support for Cohort

For Cohort, “probability” always means **P(1)**: the probability that the positive class is `1`.

Architectures that expose valid P(1) may use Cohort's probability-weighted aggregation path. Architectures that do not expose valid P(1) use Cohort's majority-vote fallback path instead.

| Architecture | Returns probabilities P(1) | Cohort mode | Notes |
|---|---:|---|---|
| `LogRegBinary` | yes | probability | `predict()` returns `_probs = predict_proba(x_test)[:, 1]`, which is directly the class-1 probability P(1). |
| `LightGBMBinary` | yes | probability | `predict()` returns `_probs` from the classifier or fitted calibrator. |
| `RandomBinary` | yes | probability | `predict()` returns `_probs`, but they are fixed surrogate scores (`0.9` for predicted 1, `0.1` for predicted 0), not model-derived calibrated probabilities. Still usable as P(1)-shaped output if Cohort accepts implementation-defined probability-like outputs. |
| `TabPFNBinary` | yes | probability | `predict()` returns `_probs` as positive-class probability. When a `CalibrationConfig` is configured, probabilities are optionally recalibrated and the threshold optimized before `_preds` are produced. This is compatible with P(1). |
| `RidgeRegressor` | no | fallback | Continuous `_preds` only; no class probabilities. Ordinary unqualified regressor Cohorts vote up at `prediction > 0`; qualified folds keep their separate contract. |
| `XGBoostRegressor` | no | fallback | `predict()` returns only `_preds` and does not expose `_probs`. Since this is a regressor, any Cohort use would have to fall back unless a separate binary-probability wrapper is introduced. |
| `DLinearRegressor` | no | fallback | `predict()` returns only `_preds`. Same regressor caveat as `XGBoostRegressor`. |
| `RuleBasedStrategy` | no | fallback | `predict()` returns positions as `_preds`; `_probs` is intentionally absent. |

## `predict()` Versus `evaluate()`

`predict()` is the small inference surface. For the built-in binary models it always returns:

- `_preds`
- `_probs`

When a `CalibrationConfig` is configured on the manifest and injected into the architecture, `predict()` additionally returns:

- `optimal_threshold` — the threshold chosen by the optimizer (or `0.5` when only probability calibration is configured)
- `val_score` — the metric score at that threshold; `None` when no threshold function is set

`evaluate()` passes these keys through into the results dict alongside the standard binary metrics.

`evaluate()` is the richer offline evaluation surface.

### `inline_metrics=False`

With `inline_metrics=False`, `evaluate()` omits inline confusion and backtest metrics. The returned dictionary still includes the five task metrics and private or schema-stability fields required by the architecture.

For `LogRegBinary`, the task metrics are:

- `accuracy`
- `auc`
- `precision`
- `recall`
- `fpr`

`LogRegBinary` also returns `_preds`, `optimal_threshold`, and `val_score`; the last two are `None` when no calibration or threshold step produced them. Other architectures may retain their own private prediction payloads.

### `inline_metrics=True`

With `inline_metrics=True`, `evaluate()` adds:

- `confusion_*` metrics
- `confusion_*_mean_return_pct` and `backtest_*` metrics when `price_data_for_backtest` is present

When `price_data_for_backtest` is absent, default inline evaluation still returns task metrics and confusion counts; price-derived confusion-return and backtest metrics are skipped.

For `LogRegBinary`, inline evaluation can add keys such as:

- `backtest_edge_bps_p50`
- `backtest_pnl_bps_p50`
- `backtest_cvar_95_pnl_bps`
- `confusion_tp`
- `confusion_fp`
- `confusion_precision`

That is why the reference-architecture layer is the bridge between raw model output and the experiment-level analytics surfaces.

## Class-based usage in the project

<!-- example: architecture-native-class source=docs/examples/compiled-research/family_ridge_regressor.yaml region=declaration scenario=search interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: book_ridge_regressor
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_15m_20250101_20250531.parquet
    split_dates:
      train_start: '2025-01-01'
      train_end: '2025-01-15'
      val_start: '2025-01-15'
      val_end: '2025-01-18'
      test_start: '2025-01-18'
      test_end: '2025-01-22'
    indicators:
    - func: limen.indicators.window_return
      params:
        period: 1
    target:
      name: next_return
      class: limen.targets.NextReturnTarget
      transform_params:
        periods: 1
        scale: 100.0
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
    scaler:
      class: limen.scalers.RobustScaler
    reference_architecture: limen.sfd.reference_architecture.ridge_regressor
    features:
    - func: limen.features.lag_range
      params:
        col: ret_1
        start: 0
        end: 5
  params:
    alpha:
    - 1.0
    solver:
    - svd
uel:
  output_path: compiled-family_ridge_regressor
  n_permutations: 1
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
```

This complete YAML defines the recorded Ridge family case. The CLI fits its declared training population and retains the predictions used to recompute RMSE independently. The selection scenario separately proves CLI reconstruction; training a family case does not by itself prove its reconstruction.

## Function Wrappers Versus Classes

Foundational manifests call a function wrapper through `with_reference_architecture`. The current Ridge wrapper signature is:

<!-- reference: architecture-wrapper source=limen/sfd/reference_architecture/ridge_regressor.py region=signature:ridge_regressor role=signature -->
```python
def ridge_regressor(data: dict[str, object], alpha: float=1.0, fit_intercept: bool=True, copy_X: bool=True, max_iter: int | None=None, tol: float=0.0001, solver: str='auto', positive: bool=False, random_state: int | None=42) -> dict[str, object]:
    ...
```

That wrapper:

1. instantiates the matching class
2. trains it
3. evaluates it with `inline_metrics=True`

The class is the canonical reusable architecture surface. The function wrapper is the manifest-facing adapter.

## Trainer Relationship

`Trainer` reconstructs the compiled manifest, reruns `prepare_data()` and `run_model()`, and uses the returned `_model` after metric validation.

The class-based layer matters even when daily work touches foundational SFDs:

- foundational SFDs package the experiment
- reference architecture owns the model contract
- `Trainer` replays selected rounds into validated class-based models

Deterministic models use near-exact metric tolerance; stochastic models use a wider relative tolerance and can still raise `ReconstructionError` when a replay diverges too far.

## `DLinearRegressor`

`DLinearRegressor` is Limen's canonical DLinear surface (Zeng et al. 2023, *Are Transformers Effective for Time Series Forecasting?*). It owns the DLinear semantics that downstream parity checks compare against, so every semantic choice is explicit:

- **Decomposition** — each lookback window is split by a centered moving average with replicate edge padding (the official DLinear `moving_avg` behavior; `kernel_size` must be a positive odd integer, canonical default `25`) into a trend component and a remainder component.
- **Heads** — each component gets its own linear head and the head outputs are summed, which is exactly the DLinear function class.
- **Fit** — the ridge-regularized DLinear MSE objective is minimized in closed form by SVD (`alpha` is the ridge strength; `alpha=0` gives the min-norm least-squares solution). There is no optimizer loop and no seed: `deterministic = True`, and re-running the same config reproduces the experiment log exactly on a fixed environment. Cross-BLAS-build float differences remain possible, as with any LAPACK-backed path.
- **Lookback** — the window is read from feature columns named `ret_1_lag_{i}`, ordered by lag descending so each row is the window in time order (oldest bar first). Window length is a manifest concern (`lag_range` with `end=lookback_end`); all other feature columns are ignored.
- **Target shape** — scalar per bar: `NextReturnTarget(periods=horizon, scale=100.0)`, the percentage return over the next `horizon` bars.
- **Direction mapping** — `evaluate()` mirrors `XGBoostRegressor`: `preds > 0` becomes the long/flat signal for the inline confusion and backtest metrics.

The Compiled project's recorded family case checks its prediction/target
alignment and recomputes RMSE. Separate component tests cover reconstruction. Those assertions
establish the declared case; they do not establish predictive edge or transfer a
historical performance number to a new window.

## `RuleBasedStrategy`

`RuleBasedStrategy` is the reference architecture for rule-based SFDs. It differs from the ML models in two important ways:

- `train()` is a no-op — rule-based strategies have no learnable parameters
- `evaluate()` runs across all three splits (train/val/test) and returns cross-split stability metrics in addition to per-split backtest metrics

It expects a `data_dict` produced by a manifest configured with `with_strategy()`:

| Prepared key | Rule-based input |
|---|---|
| `train`, `val`, `test` | Prepared frames containing predicate columns. |
| `strategy.conditions` | Resolved condition definitions. |
| `strategy.entry` | The root condition ID to evaluate. |

The strategy walks the boolean logic tree defined in `strategy['conditions']`, resolves each leaf condition by reading its pre-computed boolean column from the split DataFrame, and folds compound conditions via AND/OR/NOT. Positions are 0 (flat) or 1 (long) per bar.

`rule_based()` also returns its trained native model as `_model`. The model captures the selected strategy and leaf-column order and accepts a finite binary `x_test` matrix in that order for Sensor/fold inference; the existing named `test`/`strategy` route remains available. It declares `inference_output = ('binary', False)`. See [selected rule reconstruction](Trainer.md#selected-rule-rounds) for the raw-history and causality contract.

### Metrics produced

`evaluate()` returns a flat dict with three tiers:

- **Tier 1** — position stats: `num_trades_{split}`, `position_rate_{split}`
- **Tier 2** — per-split backtest: all `backtest_snapshot` output columns plus rule-only `pnl_per_trade_bps` and `num_executed_trades`, each suffixed with `_{split}` (e.g. `pnl_bps_p50_train`, `drawdown_bps_p5_test`, `pnl_per_trade_bps_test`)
- **Tier 3** — cross-split diagnostics: `drawdown_std_bps`, `is_stable`

`is_stable` is `False` until a replacement stability rule is defined against the new decoder-level ledger.

`pnl_per_trade_bps` is intentionally outside the generic 20-column snapshot. `RuleBasedStrategy` derives it from contiguous executed-position segments returned by the shared long-flat strategy, reports their count as `num_executed_trades`, compounds net per-bar returns after costs and notional sizing within each segment, then takes the arithmetic mean. No executed trades yields a zero count and `NaN` mean.

**NOTE:** Tier 2 and Tier 3 metrics require `open` and `close` columns in the split DataFrames to run the backtest. When those columns are absent, per-split backtest results are empty and `drawdown_std_bps` is returned as `None` with `is_stable` falling back to `False`.

`_preds` is present; `_probs` is intentionally absent (not applicable to rule-based strategies).

## TabPFN controls

`tabpfn_binary` exposes the nondeprecated constructor controls of
[TabPFN 6.4.1](https://github.com/PriorLabs/TabPFN/blob/v6.4.1/src/tabpfn/classifier.py).
The existing `n_ensemble_configurations` name maps to upstream `n_estimators`;
`n_preprocessing_jobs` replaces upstream's deprecated `n_jobs`. Existing positional
arguments retain their order; additional controls are keyword-only.

| Control | Wrapper default | Meaning |
|---|---|---|
| `n_ensemble_configurations` | `4` | Ensemble forward passes |
| `device` | `'cpu'` | Upstream device specification |
| `categorical_features_indices` | `None` | Categorical feature column indices |
| `softmax_temperature` | `0.9` | Probability temperature |
| `balance_probabilities` | `False` | Adjust probabilities for class frequency |
| `average_before_softmax` | `False` | Aggregate ensemble logits before softmax |
| `model_path` | `'tabpfn-v2-classifier-v2_default.ckpt'` | Checkpoint path or upstream Python model specifications |
| `ignore_pretraining_limits` | `True` | Permit inputs outside upstream pretraining limits |
| `inference_precision` | `'auto'` | Automatic precision, autocast or an upstream Python dtype |
| `fit_mode` | `'fit_preprocessors'` | Upstream fitting/cache mode |
| `memory_saving_mode` | `'auto'` | Upstream memory-saving policy |
| `random_state` | `0` | Seed, NumPy random generator/state or `None` |
| `n_preprocessing_jobs` | `1` | Preprocessing parallelism |
| `inference_config` | `None` | Expert inference configuration |
| `differentiable_input` | `False` | Upstream differentiable-input mode |
| `eval_metric` | `None` | Upstream evaluation metric |
| `tuning_config` | `None` | Upstream classifier tuning configuration |

Limen's `prediction_calibration_config` remains separate from upstream tuning.
Prediction/calibration logic and the default CPU/v2 checkpoint are preserved.
Controls pass through without coercion; upstream validates compatible combinations.
Python can supply supported upstream objects directly. YAML retains its existing
serializable-value boundary; represent a list or configuration as one candidate
inside the parameter's outer search list.

The foundational SFD searches ensemble counts `[4, 8]`, temperatures
`[0.7, 0.9, 1.1]`, and both probability-balancing/logit-averaging settings.
Other controls have singleton defaults, including `device: ['cpu']`.
Existing target and indicator domains remain. Select a bounded run explicitly:
the complete Cartesian product is large and these starting domains do not imply
better market performance. The [real-model procedure](Built-In-SFDs.md#tabpfn_binary)
verifies the supported surface and selected nondefault controls on recorded bars.

## Optional Dependencies

- `xgboost_regressor` requires `xgboost`
- `dlinear_regressor` requires `scipy` (the `stats` extra), loaded lazily inside the model
- `tabpfn_binary` requires `tabpfn`

Optional dependencies are checked when the relevant model executes, not when the package is imported.

## Read Next

- Continue to [Built-In SFDs](Built-In-SFDs.md) to see how the shipped foundational SFDs package these model surfaces.
- Continue to [Trainer](Trainer.md) for the reconstruction workflow that replays and validates selected rounds.
- Continue to [Standard Metrics Library](Standard-Metrics-Library.md) for the low-level metric helpers used inside these model classes.

For signed sizing, elapsed exits, recorded execution clocks and funding configuration, see [Backtest](Backtest.md#signed-exposure-elapsed-exits-and-funding).

### Primary-preserving meta allocation

The compiled manifest's `with_allocation(method, params)` adds a research allocator around its original primary. The native method is `limen.sfd.reference_architecture.meta_allocation.meta_allocation`; a second allocation declaration rejects. Factory callables follow `direction_sizing`'s source-bound estimator pattern. See [Backtest](Backtest.md#learned-allocation-around-a-primary) for fitting, cost and exposure semantics.

Prepared data exposes `_allocation_batches['val']` and `['test']`. The fitted model accepts `model.predict({'allocation_batch': batch})` without targets or outcomes. `AllocationBatch` binds full partition bounds, original ordered source IDs and availability, target-free primary inputs, named numeric meta features and decision-time stop/readiness values. Reordered, changed or truncated batches reject. This is not an incremental future-window inference protocol.

## `GaussianHMMBinary`

Install `vaquum-limen[hmm]`. The matching `gaussian_hmm_binary` function fits
hmmlearn's homogeneous univariate GaussianHMM on chronological training
`ret_1_lag_0` values. Inputs require contiguous `ret_1_lag_0..N` columns;
other feature columns are ignored. Use raw fractional returns, without a
per-lag scaler. The adapter converts training and inference to percentage
returns consistently, so Gaussian priors and covariance controls use those units.

Each prediction uses one completed lookback window, oldest first, restarting
from the fitted initial-state distribution. Only the terminal posterior is
retained: multiply it by the transition matrix and Gaussian positive-tail
probabilities to obtain P(next return > 0). Decisions use probability > 0.5.
This window-local filtering is causal and has identical row/batch behavior;
it deliberately differs from maintaining an online full-history state.
State IDs have no directional meaning. Labels do not fit the unsupervised model.

The function exposes 14 native controls for states, covariance, priors,
initial seed, EM iteration/tolerance/verbosity and log/scaling implementation.
`n_iter` caps EM; termination at the cap does not establish tolerance convergence.
The fitted backend retains its diagnostics. Supervised calibration is not part
of this variant. The [Bitcoin HMM study](https://arxiv.org/abs/2011.03741)
studies richer Bayesian/nonhomogeneous alternatives, rather than establishing
performance for this narrower EM implementation.

A recorded hourly-return CPU probe with two states, 24 lags, 30 EM iterations
and one thread measured median complete-cycle costs of 0.103/0.087s for
HMM/LogReg at 384 bars, 0.095/0.082s at 2048 and 0.264/0.098s at 8192.
Three alternating repeats included fresh preparation, fit and test prediction.
The larger HMM runs hit the iteration cap. These are configuration-specific
cost observations on macOS/Python 3.12.12, NumPy 2.2.6, sklearn 1.7.2,
Polars 1.42.1 and hmmlearn 0.3.3, with no efficacy or profitability claim.

## RBF SVM binary classifier

`RBFSVMBinary` / `rbf_svm_binary` fits sklearn SVC with `kernel='rbf'` and
`probability=True`. It requires both numeric training classes 0 and 1, returns
binary predictions and class-1 probabilities, and supports the standard optional
validation calibration and threshold selection. Calibration is fitted once on
validation data and retained for later target-free inference; refitting clears it.

The public controls keep sklearn meanings: `C=1.0`, `gamma='scale'`,
`shrinking=True`, `tol=0.001`, `cache_size=200` MiB, `class_weight=None`,
`verbose=False`, `max_iter=-1` and `random_state=42`. Gamma also accepts `'auto'`
or a nonnegative float. C is positive; max_iter=-1 removes the iteration limit.
Backend validation and convergence warnings remain visible. Limen predicts 1
above a class-1 probability of 0.5 unless validation calibration/thresholding is
configured. SVC's native margin-based class prediction can differ; this adapter
uses the probability decision rule so recorded probabilities reconstruct its decisions.

The [official SVC contract](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
describes at-least-quadratic fit scaling and extra probability-fit cost. A local
one-thread recorded hourly Bitcoin probe with 23 features measured median cycle
time 5.995s versus LogReg 0.219s at 5931 training rows (three repetitions; includes
preparation, fit and prediction). This configuration-specific cost is not a
performance or trading-benefit claim. [The research record](https://github.com/Vaquum/Limen/issues/297#issuecomment-6470247394)
contains the smaller windows and configuration.

The matching foundational SFD reuses LogReg preparation with SVM control domains
and bounded native data requests. The shipped `rbf_svm_binary` template supplies
every public control and a small dated experiment. The canonical compiled example
is `docs/examples/compiled-research/family_rbf_svm_binary.yaml`.

## `LSTMBinary`

Install `vaquum-limen[neural]`. `lstm_binary` fits a PyTorch LSTM and a scalar
linear head using binary cross-entropy with logits and Adam on CPU. Each input
is a completed contiguous `ret_1_lag_0..N` window ordered oldest first. Other
feature columns are ignored. One shared mean/std is fitted to training windows
only, retained, and used unchanged in target-free inference. Hidden state resets
for each window. Bidirectional mode combines both final-layer final states;
all observations in either direction are already available at the decision.

Inference uses eval mode, sigmoid probabilities and probability >0.5 decisions.
Initialization and training restore the caller's CPU RNG state. Exposed controls
cover the native LSTM structure, Adam configuration, epochs, minibatch size and
seed. Supervised validation calibration and GPU execution are outside this variant.
The [Bitcoin LSTM direction study](https://doi.org/10.1109/PDP2018.2018.00060)
establishes application relevance, without demonstrating efficacy for this
specific return-window configuration. The bundled Apple Silicon research lock
requires macOS 14 or later for its pinned PyTorch wheel. The [native LSTM contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)
defines the sequence and final-state semantics.

## `EntropyVaRRegressor`

`entropy_var_regressor` fits a lower conditional return quantile using sklearn
QuantileRegressor. For each completed raw `ret_1_lag_0..N` window it computes
binary Shannon entropy of the negative/nonnegative return fractions. Zero terms
contribute zero. Unrelated feature columns are ignored. Fit uses observed training
returns only; inference retains the coefficients and requires no target labels.

Output is a signed return quantile in the target's units. The foundational SFD
uses 24 completed hourly returns and the following 24-hour simple percentage
return; the compiled example uses 96 quarter-hour returns and a 96-bar horizon.
It reports pinball loss and strict violation rate (observed return below forecast),
plus descriptive continuous metrics. It produces no direction, probability or
trading metrics. Fitted scaler/PCA and execution policies are rejected; raw signs
must retain their meaning. Manager owns downstream decisions and execution.

Controls expose quantile (strictly between 0 and 0.5), alpha, fit_intercept,
solver and solver_options, plus SFD lookback/horizon. Defaults are quantile0.05,
alpha0, intercept enabled and HiGHS. Solver validation and warnings stay visible.
The [Bitcoin entropy-VaR study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7514585/)
motivates the sign-entropy predictor. This rolling-window, simple-return,
frozen-fit variant does not replicate its calendar-day log-return and rolling
coefficient protocol. Overlapping horizons imply dependent observations;
violation rate alone establishes neither calibrated tails nor GARCH superiority.
