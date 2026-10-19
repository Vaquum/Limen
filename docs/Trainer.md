# Trainer

New runs use compressed round artifacts; existing JSONL runs remain readable. See the [round storage format](Universal-Experiment-Loop.md#round-storage-format).
For [direction/sizing models](Reference-Architecture.md#direction-and-sizing), reconstruction checks frozen execution and learning identities before metric tolerance. Sensor retains signed size and actual decision availability. Altered source, factory, seed or execution assumptions require a new experiment.

`Trainer` is Limen's reconstruction layer for finished experiment rounds. It takes a completed saved YAML experiment directory, reconstructs the manifest and round parameters, replays selected permutations, validates their metrics, and wraps the replayed models in reusable `Sensor` objects.

Trainer bridges a selected experiment row and a trained model object for downstream inference.

## What Trainer needs

Trainer works only with YAML-based artifact directories, normally created by `limen run`, that include the metadata needed to reconstruct the manifest.

At minimum, the directory must contain:

- `metadata.json`
- `round_data.jsonl.gz`

Every nonblank `round_data.jsonl.gz` line must be a valid JSON object with `round_id` and object-valued `round_params`. Trainer rejects malformed JSONL instead of skipping corrupt lines, so promotion cannot proceed from a partial artifact view.

If `results.csv` is also present, Trainer validates replayed metrics against matching numeric columns in the original experiment log. If it is missing, Trainer skips validation and proceeds directly to creating sensors.

Metric validation is intersection-based: Trainer compares numeric, non-private metrics returned by replay only when the same key exists in `results.csv`. Parameter columns, private artifact keys, non-numeric values, and metrics absent from the original log are skipped. A requested permutation ID missing from `results.csv` remains a validation failure.

## Prerequisites

- the experiment must have a result directory created by `limen run`
- the experiment must be YAML-based and include `metadata.json["yaml_reference"]`
- the SFD must be a manifest-driven ML architecture (rule-based architectures are not supported)

## Workflow

Start from an existing experiment directory:

```python
import polars as pl
from limen.inference import Trainer

# 1) identify permutation IDs from results.csv
results = pl.read_csv('path/to/experiment/results.csv')
top_ids = results.sort('accuracy', descending=True).head(3)['id'].to_list()

# 2) replay selected permutations into Sensor objects
trainer = Trainer('path/to/experiment')
sensors = trainer.train(top_ids)

# 3) run inference on live klines
sensor = sensors[0]
bar_pred = sensor.predict(raw_klines)
```

In this workflow:

- `Trainer` reconstructs the manifest and round metadata
- `trainer.train(top_ids)` replays and validates the selected rounds
- each returned `Sensor` wraps one trained `ReferenceModel`

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

If validation fails, Trainer raises `ReconstructionError`.

```python
from limen import ReconstructionError

try:
    sensors = trainer.train(top_ids)
except ReconstructionError as e:
    print(e)
```

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

### Arguments

| Argument | Meaning |
|---|---|
| `experiment_dir` | path to the completed experiment directory |
| `data` | optional dataframe override; if omitted, Trainer fetches data from the reconstructed manifest |

Use `data=` with an exact dataframe override. Otherwise Trainer falls back to `manifest.fetch_data()`.

## `train(permutation_ids)`

```python
sensors = trainer.train(top_ids)
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

### Sensor example

```python
sensor = sensors[0]

print(sensor.permutation_id)   # 'sha256:a3f1c9'
print(sensor.manifest_id)      # 'sha256:7c2e41'
print(sensor.round_params)

bar_pred = sensor.predict(raw_klines)
```

Sensors are also callable. `__call__` aliases `predict_all` and returns `list[BarPrediction]`, one entry per prepared decision:

```python
all_preds = sensor(raw_klines)  # list[BarPrediction]
```

### Python construction

For a retained fitted model, start with one line:

```python
sensor = Sensor.from_model(fitted, selection_cutoff=cutoff, output_id='next_bar_return')
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

`Sensor.from_python(*, fitted_state, params, infer, map_output, state_id, fitting_cutoff, selection_cutoff, output_id=None)` returns a native Sensor accepting arbitrary history objects. See the executable [recorded Python workflow](Universal-Experiment-Loop.md#direct-python-run); its fixture requires a repository checkout.

The two caller functions have these contracts:

```text
infer(history, fitted_state, params) -> (metadata_frame, model_output)
map_output(model_output) -> Sequence[(prediction, probability)]
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

```python
@dataclass
class BarPrediction:
    datetime: object
    prediction: int | float | None
    probability: float | None
    reason: str | None
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

- `limen run` creates the promotion-ready experiment directory
- Trainer turns selected rounds from that directory into sensors

## Read next

- Continue to [Reference Architecture](Reference-Architecture.md) for the class-based model contract that Trainer reconstructs and replays.
- Continue to [Cohort](Cohort.md) to bind selected sensors into an ensemble inference surface.
- Continue to [Command-Line Interface](Command-Line-Interface.md) for the YAML run layer that produces result directories.

See [Reserve and assess final observations](Final-Holdout.md) for project research exclusion, frozen fitted-selection assessment, retained scenario evidence and the limits of local exposure history.

## Recorded walk-forward folds

Use `Trainer(directory, data=recorded_source).train([permutation_id], fold=3)` to refit and validate an explicitly selected zero-based fold. The same fold applies to every requested ID. The campaign must have retained `uel.recording.fold_details: true` and version 1 promotion witnesses. Historical lean records cannot be backfilled. Pin hosted source bounds or supply the original data; reconstruction rejects changed source values, selector output, split geometry, validation ratios, retained membership, exact bindings and selected fold metrics. It never validates against aggregate results.csv.

This is a validated refit, not restoration of saved weights. Optional recorded prediction vectors are observations, not a promise of identical historical stochastic weights. Native prediction must be stable across repeated and row/batch calls. Model output meaning comes from the optional immutable class declaration `inference_output = ('binary', True)`, `('continuous', False)` or `('target_exposure', False)`; unknown meaning still permits stable scalar Sensor inference. Necessary missing model, clock or row-membership evidence fails visibly.

The returned Sensor exposes read-only `fold`, `n_folds`, `provenance_digest`, `fitting_cutoff`, `selection_cutoff` and a deep-copied `fold_provenance`. The selection cutoff is the maximum timestamp in the original campaign source plus its native interval, covering competitors and selector information. Every earlier timestamp is guarded, including gaps and warm-up rows. Native input requires ordered unique nonnull UTC timestamps on the recorded grid; whole missing bars are allowed. Invalid clocks fail before inference recovery. Formed-bar failures preserve the actual formed decision clock once established. Ordinary and Python inference retain their contracts. Fold composition remains unavailable through legacy Cohort and Cohort.from_python; qualified compiled composition is a separate slice.
