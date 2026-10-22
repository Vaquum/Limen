# Carry selected behavior into inference

A Sensor makes fitted behavior available for later decisions. The [Python project](examples/python-research/README.md) wraps its retained evaluated models directly. The [Compiled project](examples/compiled-research/README.md) reconstructs selected recorded rounds through Trainer. These paths preserve different kinds of identity and evidence; Compiled Research uses YAML declarations and CLI commands.
For [direction/sizing models](Reference-Architecture.md#direction-and-sizing), reconstruction checks frozen execution and learning identities before metric tolerance. Sensor retains signed size and actual decision availability. Altered source, factory, seed or execution assumptions require a new experiment.

`Trainer` is Limen's reconstruction layer for finished experiment rounds. It takes a completed saved YAML experiment directory, reconstructs the manifest and round parameters, replays selected permutations, validates their metrics, and wraps the replayed models in reusable `Sensor` objects.

## What Trainer needs

Trainer works only with YAML-based artifact directories, normally created by `limen run`, that include the metadata needed to reconstruct the manifest.

At minimum, the directory must contain:

- `metadata.json`
- `round_data.jsonl.gz`

Every nonblank `round_data.jsonl.gz` line must be a valid JSON object with `round_id` and object-valued `round_params`. Trainer rejects malformed JSONL instead of skipping corrupt lines, so promotion cannot proceed from a partial artifact view.

If `results.csv` is also present, Trainer validates replayed metrics against matching numeric columns in the original experiment log. If it is missing, Trainer skips validation and proceeds directly to creating sensors.

Metric validation is intersection-based: Trainer compares numeric, non-private metrics returned by replay only when the same key exists in `results.csv`. Parameter columns, private artifact keys, non-numeric values, and metrics absent from the original log are skipped. A requested permutation ID missing from `results.csv` remains a validation failure.

## Prerequisites

- a saved Compiled experiment directory created by `limen run`
- the experiment must be YAML-based and include `metadata.json["yaml_reference"]`
- the SFD must use a reconstructable Manifest architecture that returns its native model; ML and rule-based architectures are supported

## Workflow

The Compiled project runs the selection YAML and calls `limen assess` with its result directory. The command applies the YAML's explicit validation ranking, reconstructs and validates the selected Sensors, then binds their Cohort. The Python signatures below are the reconstruction API reference.

<!-- example: trainer-compiled-selection source=docs/examples/compiled-research/run.sh region=selection scenario=selection interface=compiled role=executed -->
```bash
limen validate compiled-research/selection.yaml
limen profile compiled-research/selection.yaml
limen run --dry-run compiled-research/selection.yaml
limen run --no-progress-bar compiled-research/selection.yaml
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
```

In this workflow:

- `Trainer` reconstructs the manifest and round metadata
- `trainer.train(top_ids)` replays and validates the selected rounds
- each returned `Sensor` wraps one trained `ReferenceModel`

## Selected rule rounds

A reconstructed `RuleBasedStrategy` owns a copy of the selected strategy and its ordered leaf-predicate inputs. It emits binary positions without probabilities. Reconstruction replays the original metrics; live prediction evaluates the stored rule without fitting it again. Ordinary recorded rounds and currently qualified walk-forward folds use the same Trainer routes and evidence requirements as other native models.

Use the same [recorded selection workflow](#workflow) for rule-based YAML. Component authors can use the Trainer and Sensor API signatures below directly; these optional API calls are outside the Compiled authoring workflow.

Pass the completed rule experiment directory, its selected round ID, and recorded or fresh raw bars with the original source schema. Default artifacts suffice for ordinary reconstruction; a fold still requires its original opt-in promotion evidence. `train([id], fold=k)` never manufactures missing evidence.

Rule preparation computes features, retains feature-valid rows, and then evaluates predicates on those rows. Sensor preparation follows that order and rejoins the original decision clock: leading missing feature rows remain warm-up, later missing rows remain null-feature results, and predicate nulls become False before AND/OR/NOT. Active bar formation uses the caller's supplied raw history; an already formed source remains an already formed source.

Supply the same raw split or window when comparing research and inference. Rolling predicates, persistence, recency, feature warm-up and stateful common strategies depend on that window. No hidden history is fetched. Arbitrary SQL may use whole-window aggregates or future rows; replay and same-window agreement do not certify append-only causality. The caller owns causal feature and rule definitions. Historical guards and trade-policy availability/digests retain their existing meaning.

## Why Trainer Exists

Trainer deliberately preserves the original manifest's train/validation/test split. That makes its metric comparison meaningful: the replayed round uses the same preparation and evaluation path as the logged round.

Trainer handles reconstruction by:

- reconstructing the original experiment logic from `yaml_reference`
- validating that the pipeline still reproduces the logged round metrics
- wrapping the validated training-split model in a `Sensor` ready for inference

Trainer does not clone the manifest with `split_config=(1, 0, 0)` and does not fit on all available data. A caller that needs a separate all-data fit must build and validate that workflow explicitly; it will not reproduce the original test metrics.

## Training and validation

Trainer reruns:

- `manifest.prepare_data(data, round_params)`
- `manifest.run_model(prepared, round_params)`

with the original round parameters and wraps the resulting model in a `Sensor`. If `results.csv` is present, it also compares the resulting metrics against matching numeric columns in the original experiment log.

Deterministic models require near-exact metric matches. Stochastic models use a scaled tolerance.

Among compared numeric values, NaN matches only NaN and infinities match only with the same sign. Other finite/non-finite transitions fail before Sensor construction. This applies only to operands that pass the existing numeric guards; a metric column inferred as text while loading CSV remains skipped.

If validation fails, Trainer raises `ReconstructionError`.

Catch `limen.ReconstructionError` at the caller boundary when reconstruction failure needs explicit handling. A failed replay must remain visible.

This is Limen's guard against pipeline drift.

## Deterministic versus stochastic models

Trainer uses the model class's `deterministic` attribute to choose the validation tolerance.

| Model type | Validation style |
|---|---|
| `deterministic = True` | near-exact metric match |
| `deterministic = False` | scaled tolerance for expected randomness |

This is why promotion is more reliable for deterministic reference models than for intentionally stochastic ones.

## Trainer and the reference-architecture contract

Trainer does not promote arbitrary model objects. The compiled manifest resolves the original `ReferenceModel` path, and Trainer uses the model returned in `result['_model']` after replay.

That means the promotion stack depends on the [Reference Architecture](Reference-Architecture.md) contract:

- `train(data, **params)`
- `predict(data)`
- `evaluate(data, inline_metrics=True)`
- `deterministic`

## `Trainer(experiment_dir, data=None)`

<!-- reference: trainer-constructor-api source=limen/inference/trainer.py region=signature:Trainer.__init__ role=signature -->
```python
def __init__(self, experiment_dir: str | Path, data: pl.DataFrame | None=None, *, project_root: str | Path | None=None) -> None:
    ...
```

### Arguments

| Argument | Meaning |
|---|---|
| `experiment_dir` | path to the completed experiment directory |
| `data` | optional dataframe override; if omitted, Trainer fetches data from the reconstructed manifest |

Use `data=` with an exact dataframe override. Otherwise Trainer falls back to `manifest.fetch_data()`.

## `train(permutation_ids)`

<!-- reference: trainer-train-api source=limen/inference/trainer.py region=signature:Trainer.train role=signature -->
```python
@_holdout.train_guard
def train(self, permutation_ids: list[str], *, fold: int | None=None) -> list[Sensor]:
    ...
```

`permutation_ids` is a `list[str]` of SHA-256 round IDs from `round_data.jsonl.gz`. These are the `id` values in `results.csv` — use that file to identify which rounds to promote.

This method:

- verifies that the requested permutation IDs exist in `round_data.jsonl.gz`
- validates them against `results.csv` when available
- replays them with the original manifest split and round parameters
- returns `list[Sensor]`

Raises:

- `ValueError` if a permutation ID is missing
- `ReconstructionError` if validation detects metric drift

## Sensor

A `Sensor` wraps fitted state for inference. Trainer constructs compiled Sensors from saved YAML experiments; `Sensor.from_model` wraps conventional fitted models; `Sensor.from_python` accepts arbitrary live Python state.

Each compiled Sensor exposes:

- `permutation_id` — SHA-256 round ID from the experiment log; required for cohort binding
- `manifest_id` — SHA-256 content hash of the YAML manifest, carried from `metadata.json` for traceability
- `round_params` — parameter values used for this permutation

### Sensor inference API

<!-- reference: trainer-sensor-predict-api source=limen/inference/sensor.py region=signature:Sensor.predict role=signature -->
```python
def predict(self, raw_klines: pl.DataFrame) -> BarPrediction:
    ...
```

Sensors are also callable. `__call__` aliases `predict_all` and returns `list[BarPrediction]`, one entry per prepared decision:

<!-- reference: trainer-sensor-call-api source=limen/inference/sensor.py region=signature:Sensor.__call__ role=signature -->
```python
def __call__(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:
    ...
```

### Python construction

The Python project retains every evaluated fitted pipeline, selects within each feature window and wraps those same objects. No reconstruction is needed:

<!-- example: trainer-python-retained-state source=docs/examples/python-research/research.py region=experiment scenario=selection interface=python role=executed -->
```python
def run_research(bars, experiment_name):
    research = bars[:1440]
    space = {"window": [4, 12], "alpha": [0.1, 1.0]}
    retained = {}

    def model(data, round_params):
        fitted = fit(data["train"], round_params)
        forecasts = fitted.predict(data["validation"].select("momentum"))
        score = backtest(data["validation"], forecasts)
        retained[(round_params["window"], round_params["alpha"])] = fitted
        return {"validation_net_return": score}

    sfd = SimpleNamespace(params=lambda: space, prep=prep, model=model)
    uel = UniversalExperimentLoop(
        data=research, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space))
    )
    # An absolute filename stem writes only this CSV; no persisted round artifacts.
    uel.run(experiment_name, n_permutations=4, workers=1,
            post_processing=False, progress_bar=False)
    selected = (
        completed_log(uel).sort(
            ["window", "validation_net_return", "alpha"],
            descending=[False, True, False],
        ).unique(subset="window", keep="first", maintain_order=True)
        .select("window", "alpha").to_dicts()
    )
    sensors = [
        Sensor.from_model(retained[(p["window"], p["alpha"])], prepare=prepare_live,
                          params=p, fitting_cutoff=bars["datetime"][960],
                          selection_cutoff=bars["datetime"][1440], output_id="next_bar_return")
        for p in selected
    ]
    cohort = Cohort.from_python(sensors)
    return cohort, uel, selected, retained
```

<!-- reference: trainer-from-model-api source=limen/inference/sensor.py region=signature:Sensor.from_model role=signature -->
```python
@staticmethod
def from_model(model: StateT, *, selection_cutoff: datetime, prepare: Prepare[HistoryT] | None=None, feature_columns: Sequence[str] | None=None, params: Mapping[str, object] | None=None, predict: Predict[StateT, OutputT] | None=None, map_output: MapOutput[OutputT] | None=None, state_id: str | None=None, fitting_cutoff: datetime | None=None, output_id: str | None=None, missing: Missing='block', model_input: ModelInput='auto') -> _PythonSensor[HistoryT, StateT, ModelPairs]:
    ...
```

Pass a timestamped Polars feature frame to `sensor.predict(history)` or add
`prepare=features, params=p` to prepare completed raw history. This uses the exact
fitted object; no fit, split/label preparation or Trainer replay runs. See the
executable [recorded Python workflow](Universal-Experiment-Loop.md#direct-python-run)
for search, retained evaluated models and `Cohort.from_python(selected_sensors)`.

| `Sensor.from_model` argument | Default / contract |
|---|---|
| `model` | Required fitted object; retained by reference |
| `selection_cutoff` | Required aware datetime; decisions before it are blocked |
| `prepare` | Identity; optional `(history, params) -> pl.DataFrame` |
| `feature_columns` | Recorded `feature_names_in_`, otherwise ordered non-metadata columns; explicit sequence fixes selection/order |
| `params` | Empty shallow read-only snapshot |
| `predict` | Fitted `.predict`; optional `(model, selected_frame, params) -> output` |
| `map_output` | One numeric score per row → `(score, None)`; optional batch adapter returning scalar pairs |
| `missing` | `'block'`: null/nonfinite features become warm-up/null-features; `'model'` delegates missing values to fitted preprocessing |
| `model_input` | `'auto'`: named fitted model gets a frame, unnamed model an uncast ndarray; `'frame'`/`'array'` override the default call |
| `state_id` | Generated UUID; optional nonblank live identifier |
| `fitting_cutoff` | Selection cutoff as a provenance-only upper bound; never an exact measured endpoint |
| `output_id` | None (undeclared); supply matching explicit meanings to compose without a mapper |

Preparation must return a frame with the decision metadata below. Feature columns
retain their dtypes; custom prediction always receives the selected Polars frame.
The default adapter accepts one-dimensional ndarray, Series or sequence output,
including Boolean class labels mapped to 0/1. It never flattens dimensions or
converts target units. For missing-aware fitted pipelines use `missing='model'`.
Fit with named frames, or supply explicit columns, to keep feature selection
stable across dependency changes. There is no model-family whitelist.

Metadata validates before prediction. Declared unavailable and pre-selection rows
are excluded before prediction/mapping; default feature blocking labels leading
invalid rows warm-up and later ones null-features. Caller reasons take precedence
over computed feature reasons; the selection guard takes precedence over both.
Default preparation requires a DataFrame; custom preparation can accept arbitrary
history. Preparation runs once even when no prediction row is eligible.

#### Arbitrary Python inference

`Sensor.from_python(*, fitted_state, params, infer, map_output, state_id, fitting_cutoff, selection_cutoff, output_id=None)` returns a native Sensor accepting arbitrary history objects. The Python project's `calibration` scenario stores a fitted probability model and selected threshold in arbitrary caller-owned state; its inference adapter preserves the same binary decision meaning. The complete bundle includes its recorded inputs.

<!-- reference: trainer-from-python-api source=limen/inference/sensor.py region=signature:Sensor.from_python role=signature -->
```python
@staticmethod
def from_python(*, fitted_state: StateT, params: Mapping[str, object], infer: Infer[HistoryT, StateT, OutputT], map_output: MapOutput[OutputT], state_id: str, fitting_cutoff: datetime, selection_cutoff: datetime, output_id: str | None=None) -> _PythonSensor[HistoryT, StateT, OutputT]:
    ...
```

The two caller functions have these contracts:

<!-- reference: trainer-infer-protocol source=limen/inference/_python_sensor.py region=definition:Infer role=definition -->
```python
Infer = Callable[[HistoryT, StateT, Mapping[str, object]], tuple[pl.DataFrame, OutputT]]
```

<!-- reference: trainer-output-protocol source=limen/inference/_python_sensor.py region=definition:MapOutput role=definition -->
```python
MapOutput = Callable[[OutputT], Sequence[tuple[int | float | None, float | None]]]
```

`infer` owns preparation, prediction and any member aggregation. State stays by reference. String-keyed `params` is shallow-copied into a read-only mapping; `round_params` returns a shallow dict copy. Nested parameter values remain shared. Callers keep fitted state frozen and preparation causal; Sensor does not certify arbitrary Python methodology or call a fitting API.

Require a nonblank caller `state_id` and timezone-aware cutoffs with `fitting_cutoff <= selection_cutoff`; both normalize to UTC. Fitting precedes the exclusive fitting cutoff; fitting and selection precede the exclusive selection cutoff. The fitting cutoff records provenance. Sensor masks decisions before `selection_cutoff` with `inside-training-window`; equality is eligible. Identity, cutoffs, `origin='python'` and `prediction_mode='python'` are read-only. `permutation_id`, `manifest_id` and `trade_contract` are None.

Only decision metadata has a fixed schema:

| Column | Contract |
|---|---|
| `datetime` | Required non-null, unique, increasing Polars UTC Datetime (ms/us/ns), exactly representable as a Python datetime |
| `reason` | Optional String or all-null column: `warm-up`, `null-features` or None |
| `available_at_ns` | Optional nullable Int64 or all-null column; non-null availability must not precede its decision timestamp |

Missing optional values mean None; extra columns are ignored. Availability is a caller declaration retained on blocked and actionable rows. Every Python result has `trade_contract_digest=None`; no compiled identity or trade proof is inherited from composed members.

The adapter returns exactly one explicit pair per metadata row, including unavailable rows. Use a Sequence, excluding strings/bytes, generators and ndarray matrices. An actionable prediction must be a finite `numbers.Real` excluding bool; integral values become Python int and other real values become float. Negative and greater-than-one predictions remain unchanged in meaning. Optional probability must independently be finite and within [0,1]. Arrays, Decimal and Boolean scalars are invalid. A malformed pair or invalid aligned value yields `sensor-error` with both numeric fields None, retaining datetime and availability. Declared unavailable rows also clear both numeric fields.

Wrong construction types raise `TypeError`; invalid identities/cutoffs raise `ValueError`. Callback and adapter exceptions propagate. Invalid metadata or adapter cardinality raises `ValueError` without fabricated result rows. These Python rules leave compiled failure handling unchanged.

`predict(history)` calls inference once and returns the final `predict_all(history)` result; empty metadata gives `[]` for batch and a warm-up result for single inference. Each call uses the full batch path: O(n) metadata/result work plus caller inference cost. Callers may trim history while retaining required context. The worked example removes warm-up and forward-label boundaries independently for partition metrics; live inference uses preceding completed context instead.

Caller refitting creates a new fit. It provides no Trainer replay/`ReconstructionError` guarantee, compiled training identity or [protected final-holdout evidence](Final-Holdout.md). Use native [Cohort.from_python](Cohort.md#python-cohorts) for aligned composition with preserved unavailable results; compiled `set_members` continues rejecting Python-origin Sensors. Python Sensor persistence/reconstruction is unsupported.

### `predict(raw_klines) -> BarPrediction`

For compiled Sensors, takes a `pl.DataFrame` of raw klines with the same schema as the manifest data source. Returns a `BarPrediction` for the last bar. Python Sensors use the callback history contract above. Both paths return:

<!-- reference: trainer-bar-prediction source=limen/inference/_python_sensor.py region=signature:BarPrediction role=signature -->
```python
@dataclass
class BarPrediction:
    datetime: object
    prediction: int | float | None
    probability: float | None
    reason: PredictionReason | None
    available_at_ns: int | None = None
    trade_contract_digest: str | None = None
```

`BarPrediction` remains importable from `limen.inference` and `limen.inference.sensor`; new pickles name its defining module, `limen.inference._python_sensor`.

`reason` values:

| Value | Meaning |
|---|---|
| `None` | valid prediction; use `prediction` and `probability` |
| `'warm-up'` | not enough bars to satisfy indicator lookback |
| `'inside-training-window'` | compiled guarded split window, or Python decision before selection cutoff |
| `'null-features'` | mid-stream null feature value (data gap or transform anomaly) |
| `'sensor-error'` | compiled caught inference failure, or invalid aligned Python scalar/pair |

Compiled Sensors catch data/model failures after Manifest reconstruction and return `sensor-error`; malformed YAML still raises at reconstruction. `NotImplementedError` is raised if the manifest has `decoder_lookback > 1` (not yet supported). Python callback/adapter exceptions and malformed metadata/cardinality raise as described above.

### `predict_all(raw_klines) -> list[BarPrediction]`

Compiled Sensors return one `BarPrediction` per bar in the post-bar-formation data. Warm-up bars and inside-window bars have `reason` set. Python Sensors return one result per metadata row; probability is optional.

Compiled caught failures return `sensor-error` entries (one per prepared bar, or input bar if preparation fails). Manifest reconstruction and unsupported decoder lookback still raise. Python failure behavior follows the construction contract above.

### What `predict()` expects

Compiled Sensor callers pass raw klines, not `x_test`. The Sensor rebuilds the feature matrix internally from the stored manifest, fitted preprocessing params, and round params, then passes `x_test` to the trained reference model. Calibrated models (those trained with `use_calibration: true`) store the fitted calibrator internally during the training evaluation step; subsequent `predict()` calls reuse it without needing `x_val` or `y_val`. The caller never needs to supply validation data at inference time. Compiled guards retain date-granularity comparisons; Python cutoffs compare complete UTC datetimes.

## What Trainer reads from disk

Trainer uses:

- `metadata.json` — reads `yaml_reference` to reconstruct the manifest; `manifest_id` is also read and passed to each Sensor
- `round_data.jsonl.gz` — loads `round_params` for each permutation
- `results.csv` — when available, validates replayed metrics against matching numeric columns in the original experiment log

## Scope note

Trainer depends on a saved YAML experiment definition, normally produced by CLI. A Python/MSQ run is compatible only when its saved `yaml_reference` reconstructs the same manifest. Both experiment routes use UEL with a concrete `SearchStrategy`; arbitrary custom SFD functions alone do not provide this reconstruction contract.

The operating model is:

- a saved Compiled run preserves the reconstructible definition and configured evidence
- Trainer replays selected recorded rounds and returns Sensors
- Python Research wraps retained fitted state directly and owns any additional persistence contract

## Read next

- Continue to [Reference Architecture](Reference-Architecture.md) for the class-based model contract that Trainer reconstructs and replays.
- Continue to [Cohort](Cohort.md) to bind selected sensors into an ensemble inference surface.
- Continue to [Command-Line Interface](Command-Line-Interface.md) for the YAML run layer that produces result directories.

See [Reserve and assess final observations](Final-Holdout.md) for project research exclusion, frozen fitted-selection assessment, retained scenario evidence and the limits of local exposure history.

## Recorded walk-forward folds

Use `Trainer(directory, data=recorded_source).train([permutation_id], fold=3)` to refit and validate an explicitly selected zero-based fold. The same fold applies to every requested ID. The campaign must have retained `uel.recording.fold_details: true` and version 1 promotion witnesses. Historical lean records cannot be backfilled. Pin hosted source bounds or supply the original data; reconstruction rejects changed source values, selector output, split geometry, validation ratios, retained membership, exact bindings and selected fold metrics. It never validates against aggregate results.csv.

This is a validated refit, not restoration of saved weights. Optional recorded prediction vectors are observations, not a promise of identical historical stochastic weights. Repeated full-batch predictions must match exactly; row/batch scalar comparisons allow 1e-6 absolute and relative floating-point roundoff. Model output meaning comes from the optional immutable class declaration `inference_output = ('binary', True)`, `('continuous', False)` or `('target_exposure', False)`; unknown meaning still permits stable scalar Sensor inference. Necessary missing model, clock or row-membership evidence fails visibly.

<!-- example: trainer-compiled-folds source=docs/examples/compiled-research/run.sh region=walk_forward scenario=walk_forward interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/walk_forward.yaml
limen assess compiled-research/walk_forward.yaml --results results/dev/compiled-walk_forward
limen run --no-progress-bar compiled-research/acceptance.yaml
```

The returned Sensor exposes read-only `fold`, `n_folds`, `provenance_digest`, `fitting_cutoff`, `selection_cutoff` and a deep-copied `fold_provenance`. The selection cutoff is the maximum timestamp in the original campaign source plus its native interval, covering competitors and selector information. Every earlier timestamp is guarded, including gaps and warm-up rows. Native input requires ordered unique nonnull UTC timestamps on the recorded grid; whole missing bars are allowed. Invalid clocks fail before inference recovery. Formed-bar failures preserve the actual formed decision clock once established. Ordinary and Python inference retain their contracts. Use [qualified compiled Cohort selection](Cohort.md#recorded-walk-forward-members) to compose compatible binary folds and ordinary members. Legacy Cohort binding and Cohort.from_python still reject fold-bearing Sensors.
