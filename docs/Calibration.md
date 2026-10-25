# Calibrate probabilities and choose a decision boundary

Calibration changes probability estimates. Threshold selection changes the decision made from those estimates. Both consume validation evidence; neither supplies an untouched assessment of the selected model.

The [Python project](examples/python-research/README.md) owns these steps as ordinary Python functions. The [Compiled project](examples/compiled-research/README.md) declares them in YAML and runs the fitting and recording through the CLI. Its complete `research.yaml` also carries the selected calibrated models through `limen assess`. Both projects compare all four on/off combinations in their `calibration` scenario.

## Prerequisites

Use the complete project entry points and matching recorded input bundle. Python fits its classifier and calibrator in caller-owned state. Compiled Research runs the four calibration/threshold combinations declared in `calibration.yaml`; its CSV retains each mode and selected threshold. `calibration_assessment.yaml` adds an event-return objective, product and source clock, then assesses all four recorded modes. The assessment partition supplies no calibration or threshold fitting data.

## What Calibration Does

Binary classifiers return raw probabilities, but raw probabilities can be miscalibrated. A model output of `0.7` is not automatically a calibrated `70%` positive-class probability. Probability calibration fits a post-hoc correction on a held-out validation set so the output probabilities align with actual class frequencies.

Threshold optimization solves a different problem. A decision boundary at `0.5` need not maximize the chosen metric, particularly when false positives and false negatives have different consequences. Limen's `grid_threshold_optimizer` sweeps a range of candidate thresholds, scores each one with a chosen metric, and selects the highest-scoring candidate.

The two steps are independent and can be used together or separately.

## The Four Modes

| `use_calibration` | `use_threshold` | Behavior |
|---|---|---|
| True | True | calibrate probabilities → optimize threshold |
| True | False | calibrate probabilities → decide at 0.5 |
| False | True | raw probabilities → optimize threshold |
| False | False | raw model output, no calibration injected |

For a configured Compiled calibration, both flags default to `True`. Register the required calibrator and/or optimizer before calling `.done()`; `with_calibration()` alone does not register either step.

On ML manifests, `use_calibration` and `use_threshold` require actual Booleans wherever supplied: YAML candidate lists, native parameter domains, context overrides and effective rounds. Strings, numeric 0/1 and explicit null are rejected before preparation or fitting, including when the other control is false or no calibration block is declared. YAML Boolean anchors and aliases retain their Boolean meaning through search, saved metadata and Trainer replay. Omitting a control preserves its enabled default; a disabled control does not require registering its component.

When both are `False`, no `prediction_calibration_config` is injected into the architecture and the model behaves exactly as if calibration was never configured.

## Quick Start

Read the Python function and Compiled declaration together. Python returns its explicit threshold; the compiled classifier retains nullable calibration fields in a stable result schema.

<!-- example: calibration-python-modes source=docs/examples/python-research/scenarios.py region=calibration scenario=calibration interface=python role=executed -->
```python
def calibration(data_dir, output):
    bars = quarter_hour(data_dir)
    params = {'window': 4, 'alpha': .1}
    train, validation = project.prep(bars, params).values()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, random_state=42)).fit(train.select('momentum'), train['next_return'].to_numpy() > 0)
    modes, sensors = [], []
    for calibrate in (False, True):
        fitted = sklearn_probability_calibrator(model, validation.select('momentum'), validation['next_return'].to_numpy() > 0, method='sigmoid') if calibrate else model
        probabilities = fitted.predict_proba(validation.select('momentum'))[:, 1]
        for tune in (False, True):
            threshold, score = grid_threshold_optimizer(validation['next_return'].to_numpy() > 0, probabilities, metric=balanced_metric) if tune else (.5, None)
            state = SimpleNamespace(model=fitted, threshold=threshold)
            sensor = Sensor.from_python(fitted_state=state, params=params, infer=infer_classifier, map_output=lambda rows: rows, state_id=f'sigmoid={calibrate};threshold={tune}', fitting_cutoff=bars['datetime'][960], selection_cutoff=bars['datetime'][1440], output_id='next_bar_up')
            sensors.append(sensor)
            cohort = Cohort.from_python([sensor], aggregate='vote')
            tape = cohort.predict_all(bars[1428:1920])[12:]
            if any(point.prediction not in (0, 1) for point in tape):
                raise ValueError('Classifier inference must provide a binary decision for every assessment row')
            modes.append({'calibrated': calibrate, 'tuned': tune, 'threshold': threshold, 'validation_score': score, 'validation_probabilities': probabilities.tolist(), 'validation_labels': (validation['next_return'].to_numpy() > 0).astype(int).tolist(), 'inference_rows': len(tape), 'positive_predictions': sum(point.prediction == 1 for point in tape)})
    methods = []
    for method in ('sigmoid', 'isotonic'):
        calibrated = sklearn_probability_calibrator(model, validation.select('momentum'), validation['next_return'].to_numpy() > 0, method=method)
        probabilities = calibrated.predict_proba(validation.select('momentum'))[:, 1]
        threshold, score = grid_threshold_optimizer(validation['next_return'].to_numpy() > 0, probabilities, metric=balanced_metric)
        state = SimpleNamespace(model=calibrated, threshold=threshold)
        sensor = Sensor.from_python(fitted_state=state, params=params, infer=infer_classifier, map_output=lambda rows: rows,
            state_id=method, fitting_cutoff=bars['datetime'][960], selection_cutoff=bars['datetime'][1440], output_id='next_bar_up')
        tape = sensor.predict_all(bars[1428:1920])[12:]
        methods.append({'method': method, 'calibrated': True, 'tuned': True, 'threshold': threshold, 'validation_score': score,
            'validation_probabilities': probabilities.tolist(), 'validation_labels': (validation['next_return'].to_numpy() > 0).astype(int).tolist(),
            'inference_rows': len(tape), 'positive_predictions': sum(point.prediction == 1 for point in tape)})
    return {'probability_methods': methods, 'modes': modes, 'decisions': [{key: mode[key] for key in ('calibrated', 'tuned', 'threshold', 'inference_rows', 'positive_predictions')} for mode in modes], 'validation_reused': True, 'composition': compose_selected(sensors, bars)}
```

<!-- example: calibration-compiled-modes source=docs/examples/compiled-research/calibration.yaml region=declaration scenario=calibration interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: book_compiled_calibration
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
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: roc_1
        quantile: 0.5
      transform_params:
        shift: -1
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
    scaler:
      from_params: scaler_type
    reference_architecture: limen.sfd.reference_architecture.logreg_binary
# region calibration_config
    calibration:
      probability_calibration:
        func: limen.calibration.sklearn_probability_calibrator
        params:
          method: sigmoid
      threshold_function:
        func: limen.calibration.grid_threshold_optimizer
        params:
          metric: limen.metrics.balanced_metric.balanced_metric
# endregion calibration_config
  params:
    scaler_type: [logreg]
    C:
    - 1.0
    random_state: [42]
# region calibration_modes
    use_calibration:
    - false
    - true
    use_threshold:
    - false
    - true
# endregion calibration_modes
uel:
  output_path: compiled-calibration
  n_permutations: 4
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
```

The Python example uses sigmoid calibration, a caller-owned fitted state and `Sensor.from_python`. It wraps that Sensor in a voting Cohort and checks every assessment decision is binary. The Compiled project uses this declaration:

<!-- example: calibration-compiled-config source=docs/examples/compiled-research/calibration.yaml region=calibration_config scenario=calibration interface=compiled role=executed -->
```yaml
    calibration:
      probability_calibration:
        func: limen.calibration.sklearn_probability_calibrator
        params:
          method: sigmoid
      threshold_function:
        func: limen.calibration.grid_threshold_optimizer
        params:
          metric: limen.metrics.balanced_metric.balanced_metric
```

`sklearn_probability_calibrator` wraps `CalibratedClassifierCV` with `FrozenEstimator`, preserving the already-fitted classifier. It accepts isotonic calibration (the default) and sigmoid calibration. `grid_threshold_optimizer` chooses the highest-scoring admissible threshold; the default `balanced_metric` is `precision * sqrt(trade_rate)`.

<!-- reference: calibration-probability-api source=limen/calibration/probability.py region=signature:sklearn_probability_calibrator role=signature -->
```python
def sklearn_probability_calibrator(clf: Any, x_val: npt.NDArray[Any] | pl.DataFrame, y_val: npt.NDArray[Any] | pl.Series, method: str='isotonic') -> Any:
    ...
```

<!-- reference: calibration-threshold-api source=limen/calibration/threshold.py region=signature:grid_threshold_optimizer role=signature -->
```python
def grid_threshold_optimizer(y_val: npt.NDArray[Any] | pl.Series, val_proba: npt.NDArray[np.floating[Any]], threshold_min: float=0.2, threshold_max: float=0.7, threshold_step: float=0.05, default_threshold: float=0.35, metric: Callable[[Any, Any], float]=balanced_metric, *, _objective_maximize: bool | None=None) -> tuple[float, float]:
    ...
```

## Threshold-Only Path

The Python scenario's `calibrate=False, tune=True` case optimizes raw model probabilities. Its `calibrate=True, tune=False` case calibrates and uses `0.5`. In a Manifest, register just the threshold function for threshold-only behavior. Register just a probability calibrator for calibrated decisions at `0.5`.

The builder methods below describe the current fluent API; they are reference signatures, rather than a separate executable experiment. Obtain the builder from `MLManifest.with_calibration()` and return to the Manifest with `.done()`.

<!-- reference: calibration-builder-entry source=limen/experiment/manifest_core.py region=signature:MLManifest.with_calibration role=signature -->
```python
def with_calibration(self) -> 'CalibrationBuilder':
    ...
```

<!-- reference: calibration-builder-probability source=limen/experiment/manifest_core.py region=signature:CalibrationBuilder.probability_calibration role=signature -->
```python
def probability_calibration(self, func: CalibratorProtocol, **params: Any) -> 'CalibrationBuilder':
    ...
```

<!-- reference: calibration-builder-threshold source=limen/experiment/manifest_core.py region=signature:CalibrationBuilder.threshold_function role=signature -->
```python
def threshold_function(self, func: ThresholdOptimizerProtocol, **params: Any) -> 'CalibrationBuilder':
    ...
```

<!-- reference: calibration-builder-done source=limen/experiment/manifest_core.py region=signature:CalibrationBuilder.done role=signature -->
```python
def done(self) -> 'MLManifest':
    ...
```

## Searching Over Calibration Parameters

Manifest builder string parameters name round-parameter references. For example, registering `method='cal_method'` resolves the current round's `cal_method`; non-string floats and callables pass through unchanged. YAML uses explicit parameter references in the declared component parameters. Search domains and the admissibility of their combinations remain part of the experiment definition.

## On/Off Comparison Within a Single Experiment

The Compiled project declares its 2×2 comparison here:

<!-- example: calibration-compiled-flags source=docs/examples/compiled-research/calibration.yaml region=calibration_modes scenario=calibration interface=compiled role=executed -->
```yaml
    use_calibration:
    - false
    - true
    use_threshold:
    - false
    - true
```

`run_model()` consumes these flags before calling the architecture. The architecture receives the resulting, possibly masked `CalibrationConfig`. The four rows expose each step's effect within this recorded experiment; comparing them still reuses validation evidence.

## Custom Calibrators and Optimizers

A custom probability calibrator accepts the fitted classifier and validation features/labels, and returns an object with `predict_proba()`. A custom optimizer accepts validation labels/probabilities and returns a threshold and score. The shared protocols are:

<!-- reference: calibration-calibrator-protocol source=limen/calibration/pipeline.py region=signature:CalibratorProtocol.__call__ role=signature -->
```python
def __call__(self, clf: Any, x_val: npt.NDArray[Any] | pl.DataFrame, y_val: npt.NDArray[Any] | pl.Series, **params: Any) -> Any:
    ...
```

<!-- reference: calibration-optimizer-protocol source=limen/calibration/pipeline.py region=signature:ThresholdOptimizerProtocol.__call__ role=signature -->
```python
def __call__(self, y_val: npt.NDArray[Any] | pl.Series, val_proba: npt.NDArray[np.floating[Any]], **params: Any) -> tuple[float, float]:
    ...
```

The protocols are independent: a custom calibrator can use the built-in threshold optimizer, and a custom optimizer can use the built-in calibrator. Python owns their composition directly; Compiled Research declares resolvable functions under `calibration`. The signatures below document component extension.

## What Gets Added to Results

For the configured Compiled classifier, `evaluate()` always includes these two keys so the experiment log schema stays stable across rounds (e.g. when `use_calibration`/`use_threshold` are searched as `[True, False]`):

| Key | Type | Meaning |
|---|---|---|
| `optimal_threshold` | `float` or `None` | the threshold chosen by the optimizer (`None` when both calibration flags disable the configuration for this round) |
| `val_score` | `float` or `None` | the metric score at that threshold; `None` when no threshold function ran |

These appear in the experiment log alongside the standard binary metrics. `Trainer` validates them during reconstruction; `Sensor` does not expose an experiment-results mapping.

The lower-level `predict()` only includes these keys when calibration is active for the round. `evaluate()` always emits them to ensure every row in the CSV has a consistent schema.

`_preds` are computed using `optimal_threshold`: a test observation is predicted positive when its probability is at or above the threshold.

## `grid_threshold_optimizer` Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `threshold_min` | `0.20` | minimum threshold to test |
| `threshold_max` | `0.70` | maximum threshold to test |
| `threshold_step` | `0.05` | step size for the sweep |
| `default_threshold` | `0.35` | fallback when no threshold produces positive predictions |
| `metric` | `balanced_metric` | scoring function `(y_true, y_pred) -> float` |

The optimizer returns the `default_threshold` with score `0.0` when every candidate threshold predicts all negatives.

Threshold grids are caller-owned inputs: `threshold_step` must be positive and finite, bounds must be finite, and `threshold_min <= threshold_max`. The default-threshold fallback is for valid grids that produce no positive predictions, not a substitute for validating a malformed grid.

## Calibration at Inference

The Python scenario stores the fitted probability model and selected threshold together in its caller-owned state. Its inference function prepares completed history and reuses that state without fitting. The Compiled calibration scenario records all four modes. The complete `research.yaml` additionally declares assessment: `limen assess` reconstructs the selected Sensors with their fitted preprocessing, calibration and thresholds.

A Sensor accepts history through its native inference API. The lower-level classifier's `predict()` accepts the prepared feature matrix; these are distinct boundaries. `fit_calibrator` provides the shared fit operation outside a Manifest too.

## Where To Look

- `limen/calibration/` provides the calibration protocols, fitting, probability correction, threshold optimization and prediction helpers.
- `limen.experiment.CalibrationConfig` holds the resolved configuration injected into a Compiled architecture.
- `CalibrationBuilder` is obtained through `MLManifest.with_calibration()`; it is not a public import.

## Shared validation-return objective

The optional `sfd.manifest.objective` declares the measure for threshold search and existing metric-driven reducers:

<!-- example: calibration-compiled-objective source=docs/examples/compiled-research/selection.yaml region=validation_objective scenario=selection interface=compiled role=executed -->
```yaml
    objective:
      metric: backtest_total_return
      direction: maximize
```

Native manifests use `manifest.set_objective(metric="backtest_total_return", direction="maximize")`. Only event-ledger total return and `maximize`/`minimize` are supported. Return is ending equity divided by initial equity minus one, including configured execution costs/funding and the engine's terminal mark; it is not a closed-trade return formula. Compatible ML and rule-based architectures require aligned validation decisions and an explicit event-execution product. Fitted predictors receive validation-only inputs; equivalent wrappers and explicit bound validation evidence follow the same contract. Signed exposure and allocation objectives score their actual decisions without binary threshold tuning. Fee/slippage-only snapshot backtests are rejected. An active custom threshold optimizer receives the causal validation scorer and declared maximize/minimize direction. It must call that scorer on its selected `probabilities >= threshold` decisions and return a finite threshold/score pair agreeing with that validation return. Bound metric, direction or validation operands conflict with the objective and reject before model fitting. Disabled optimizers are not invoked. Partials, decorators and callable instances use the same contract; ordinary nonprotected call-time keywords retain their existing precedence.

The Compiled project's `selection.yaml` pairs this objective with an explicit cash-spot product and ranks its recorded rounds by validation return. Its `walk_forward` scenario also tunes thresholds against validation execution. The objective does not enable threshold search on its own: existing calibration/threshold flags still control tuning. Python Research can call the shared execution engine inside its own scorer; the Python project keeps its explicit caller-owned net-return calculation.

Threshold candidates use validation execution only. All-flat is represented by the finite threshold `2.0` with `>=`, so it remains flat at inference even for probability `1.0`. Exact ties prefer all-flat, then the higher threshold. Candidate execution failures and invalid probabilities/returns raise; there is no economic-mode fallback.

Every successful objective round reports raw `val_backtest_total_return`, including untuned rounds using the fitted model's actual decision rule. `val_score` remains the threshold scorer's score, or null when tuning is off. Held-out `backtest_*` columns retain their test meaning. Candidate scoring cannot replace test execution evidence. Missing causal validation coverage, including separate-source boundary coverage, fails instead of using held-out prices.

YAML binds Correlation, Focus, Sanity, Saturation and worst-first Budget reducers to the validation column; it sets comparison direction where applicable. Explicit matching values are allowed and conflicts fail before execution. Native reducers must be constructed with `metric="val_backtest_total_return"` and matching `maximize`. Random Budget remains metric-free. Automatic ranking means Focus best-row and Budget worst-first ordering; cohort selectors and custom/manual feedback retain their existing interfaces.

Persisted YAML/MSQ experiments save `metadata.json.objective` and reconstruct the same operating point through Trainer/Sensor. Resume accepts the same declaration and rejects changes/addition/removal or a missing validation-score column before rewriting artifacts. Legacy runs keep their existing schemas. Standard runs score without new resume/promotion support; runs without an experiment directory keep their objective results in memory.

These are validation selection scores. Calibration/search reuse validation data, tuned-versus-untuned comparisons are optimistic, and valid all-flat zero returns can tie. They establish no statistical acceptance or causal effect. Scoring cost grows with the existing threshold grid. Existing diagnostic/resource controls remain independent of the objective.

## Read Next

Continue to [search and recovery](Advanced-Search.md) to compare candidates, [the Manifest reference](Experiment-Manifest.md) for builder declarations, and [Sensor construction](Trainer.md) for fitted model reuse.

## Read the current decisions

<!-- result: calibration-python-decisions interface=python scenario=calibration fields=decisions,validation_reused -->

<!-- result: calibration-compiled-decisions interface=compiled scenario=calibration fields=modes,reconstructed_modes -->

## Both probability-calibration methods

The Python calibration variant fits both `sigmoid` and `isotonic` on the same declared validation population, then carries the fitted calibrator and chosen threshold into 480 later decisions. The Compiled `calibration_isotonic.yaml` variant uses the same four calibration/threshold switches as the sigmoid declaration and reconstructs all four selected Sensors through `limen assess`. The method changes the probability mapping; it does not provide new independent validation evidence.

### Python Research methods

<!-- result: calibration-python-methods interface=python scenario=calibration fields=probability_methods[].method,probability_methods[].threshold,probability_methods[].inference_rows,probability_methods[].positive_predictions -->

### Compiled Research methods

<!-- result: calibration-compiled-methods interface=compiled scenario=calibration fields=probability_methods[].method,probability_methods[].reconstructed_members -->
