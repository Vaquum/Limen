# Universal Experiment Loop

Universal Experiment Loop (UEL) is an integral part of Loop, and takes as its input data and a Single-File Model (SFM). `UEL` currently wraps onto itself (i.e. the object `uel.run` yields) all the folds from `Data` to `Backtest`. In other words, all the following folds are wrapped into one workflow `uel.run`:

- [`Data`](HistoricalData.md)
- [`Indicator`](Indicators.md)
- [`Feature`](Features.md)
- [`SFM`](Single-File-Model.md)
- [`UEL`](Universal-Experiment-Loop.md) 
- [`Log`](Log.md) 
- `Benchmark`
- `Backtest`

The operation of `uel.run` can be thought of as an advanced parameter sweep, which automatically integrates the parameter sweep results (i.e. `uel.experiment_log`) with benchmarks (i.e. `uel.experiment_confusion_metrics`) and backtest (i.e. `uel.experiment_backtest_results`).

## On Parameters

Parameters can be thought of as "knobs", as a means to control the direction of the experiment. Therefore, parameters play a crucial role in determining how successful an experiment is. Therefore, it is wise to start with a rich set of parameters, and broad parameter ranges for each parameter. The parameter sweep method is extremely powerful this way; in terms of its potential, it can be imagined as having infinite number of hands, turning an infinite number of knobs at the same time.

## The True Meaning of Parameter Sweep

It is typically thought that the focus of the parameter sweep is specifically the model hyperparameters, and only these. This led to the bastardized term "hyperparameter optimization". This perspective is extremely limiting and entirely misses the point of parameter sweeping. 

In short, the point of parameter sweeping is that since such a practice is possible, and since more or less anything and everything can be readily parametrized, there should be no limit to where this approach can be applied. 

**Not only the idea of sweeping through parameters can be extended beyond the model and its hyperparameters, to data fetching, data pre-processing, feature engineering, and all other aspects of classifier development lifecycle, but it can also be extended well beyond input arguments. For example, conditional logic can be handled as parameters, and even individual fragments of code can be fully parametric, and therefore a subject of a parameter sweep.**

In other words, the idea of performing a parameter sweep is equally relevant to all of Loop's folds. This is a crucial key point, and our success depends on understanding it, putting it into practice, and realizing its unrestrained power to yield the most meaningful probabilities for live trading at any given point in time, regardless of the prevailing circumstances.

## Performing an Experiment

The meaning of the term experiment is encapsulated by the below workflow. 

`Choose Data` -> `Choose Indicators` -> `Choose Features` -> `Develop SFM` -> `Run UEL` -> `Analyze Experiment Log` -> `Analyze Experiment Confusion Metrics` -> `Analyze Backtest Results` -> `Refine Parameters` -> `Run UEL` -> `...`

For an SFM to become mature and ready for trading, one must iterate between running `UEL` and refining parameters many times. Generally speaking, even a relatively small parameter space requires thousands or tens of thousands of permutation rounds before meaningful analytical power is unlocked.

## Refining Parameters

Refining parameters can be understood through expanding or contracting parameters or parameter value ranges. 

## Data

A key point here is that all individual contributors work based on the same underlying data. We achieve this by always calling data from the provided (klines) endpoints available through [HistoricalData](Historical-Data.md). If you don't find what you need through these endpoints, [make an issue](https://github.com/Vaquum/Loop/issues/new) that requests the data that you need, or make a PR that commits the proposed change. 

## SFM

An SFM contains all parameters, data preparation code, and model operation codes in a single python file. For example, representing an XGBoost bullish regime binary classifier.

Read more in [Universal Experiment Loop](Universal-Experiment-Loop.md).

## `UniversalExperimentLoop`

Initializes the Universal Experiment Loop.

### Args

| Parameter             | Type               | Description                                           |
|-----------------------|--------------------|-------------------------------------------------------|
| `data`                | `pl.DataFrame`     | The data to use for the experiment.                   |
| `single_file_model`   | `SingleFileModel`  | The single file model to use for the experiment.      |


### `run`

Runs the experiment `n_permutations` times. 

**NOTE:** You can always over-ride any or all of `params` or `prep` or `model` by passing them through here as input arguments. Just make sure that the inputs and returns are same as the ones outlined in [Single File Model](Single-File-Model.md).

### Args

| Parameter                     | Type              | Description                                                                                             |
|-------------------------------|-------------------|---------------------------------------------------------------------------------------------------------|
| `experiment_name`             | `str`             | The experiment name. Also used as the CSV filename written in the project root (`<name>.csv`).          |
| `n_permutations`              | `int`             | Number of permutations to run (default: 10000).                                                         |
| `prep_each_round`             | `bool`            | If `True`, calls `sfm.prep(data, round_params=...)` on every round. If `False`, calls `sfm.prep(data)` once on the first round. |
| `random_search`               | `bool`            | If `True`, sample the parameter space randomly; if `False`, iterate deterministically (grid-like).      |
| `maintain_details_in_params`  | `bool`            | If `True`, injects `_experiment_details` (e.g., current index) into `round_params` for logging, then removes it before writing results. |
| `context_params`              | `dict`            | Extra context passed into `round_params` for every round (useful for identifiers like symbol, horizon). |
| `save_to_sqlite`              | `bool`            | If `True`, appends results to SQLite at `/opt/experiments/experiments.sqlite` under table `experiment_name`. |
| `params`                      | `Callable`        | Optional override for the SFM `params` function. Must return a parameter space dictionary.              |
| `prep`                        | `Callable`        | Optional override for the SFM `prep` function. Must follow the standard input/output contract.          |
| `model`                       | `Callable`        | Optional override for the SFM `model` function. Must follow the standard input/output contract.         |
| `manifest`                    | `Manifest`        | Optional [Experiment Manifest](Experiment-Manifest.md) for Universal Split-First data processing. When provided, enables advanced data preparation pipelines with bar formation and fitted parameters. |

### Returns

Adds artifacts into the `UniversalExperimentLoop` instance and writes streaming logs:

- CSV logging: On the first round, writes a header row to `<experiment_name>.csv`, then appends one result row per round.
- Optional SQLite logging: If `save_to_sqlite=True`, appends the last written row to `/opt/experiments/experiments.sqlite` under table `experiment_name`.

#### Artifacts on the `UniversalExperimentLoop` instance

- `data`: The original `pl.DataFrame` passed to the loop
- `params`: The parameter space in use (from SFM or `params` override)
- `prep`, `model`: The callable functions in use (from SFM or overrides)
- `round_params`: `list[dict]` of the actual parameter sets used in each permutation
- `experiment_log`: `pl.DataFrame` containing accumulated round results (first row created on round 0, then vstacked)
- `extras`: `list[Any]` of any extra artifacts returned by the SFM (when `round_results` contained an `extras` key)
- `models`: `list[Any]` of models returned by the SFM (when `round_results` contained a `models` key)
- `preds`: `list[Any]` predictions captured from the SFM when `round_results['_preds']` is present
- `scalers`: `list[Any]` scalers captured from SFM `prep` when the data dict includes `_scaler`
- `_alignment`: `list[dict]` alignment metadata per round, as produced by `utils.splits.split_data_to_prep_output`:
  - `missing_datetimes`: list of datetimes dropped during prep
  - `first_test_datetime` / `last_test_datetime`: inclusive test window bounds
  
  In addition, UEL now exposes precomputed analysis artifacts and a convenience handle to the internal logger:
  
  - `experiment_confusion_metrics`: `pd.DataFrame` produced via `Log.experiment_confusion_metrics('price_change')`
  - `experiment_backtest_results`: `pd.DataFrame` produced via `Log.experiment_backtest_results()`
  - `experiment_parameter_correlation`: Convenience reference to `Log.experiment_parameter_correlation`
  - `_log`: Internal `Log` instance used to compute the above artifacts
  - `explorer`: Convenience callable that launches the Loop Explorer Streamlit UI bound to this UEL instance (dataset selector includes Historical Data, Experiment Log, Confusion Metrics, and Backtest Results)

#### Parameter space

UEL uses `utils.param_space.ParamSpace` to generate `round_params` either via random sampling or deterministic iteration, depending on `random_search`.

#### Log integration

At the end of `run`, UEL constructs an internal `Log` instance and exposes key analysis artifacts directly on the UEL object:

- `self._log = loop.Log(uel_object=self, cols_to_multilabel=<all string columns>)`
- Precomputed properties set on UEL:
  - `self.experiment_confusion_metrics = self._log.experiment_confusion_metrics('price_change')`
  - `self.experiment_backtest_results = self._log.experiment_backtest_results()`
  - `self.experiment_parameter_correlation = self._log.experiment_parameter_correlation`

If you need additional methods, access the internal `Log` via `uel._log` (see [Log](Log.md)):
- `permutation_prediction_performance(round_id: int) -> pd.DataFrame`
- `permutation_confusion_metrics(x: str, round_id: int, ...) -> pd.DataFrame`
- `read_from_file(file_path: str) -> pd.DataFrame` (alternative constructor path)


**NOTE**: For fully reproducible post-experiment analysis with `Log`, if using `prep_each_round=True`, make sure that `sfm.prep` is not any random operations, or if you must have random operations, use parametric seeds for round-by-round reproducibility.

### Example

```python
import loop
from loop import sfm

# Standard SFM without manifest
uel = loop.UniversalExperimentLoop(
    data=your_polars_dataframe,
    single_file_model=sfm.lightgbm.breakout_regressor,
)

uel.run(
    experiment_name='exp_breakout',
    n_permutations=10,
    prep_each_round=False,
    random_search=True,
    maintain_details_in_params=True,
    save_to_sqlite=False,
)

# SFM with manifest for Universal Split-First processing
uel_manifest = loop.UniversalExperimentLoop(
    data=your_polars_dataframe,
    single_file_model=sfm.reference.logreg,
)

manifest = sfm.reference.logreg.manifest()
uel_manifest.run(
    experiment_name='exp_logreg_manifest',
    n_permutations=100,
    prep_each_round=True,
    manifest=manifest,
)

# Post-run analysis via precomputed artifacts and internal Log
backtest_df = uel.experiment_backtest_results
confusion_df = uel.experiment_confusion_metrics

# Compute round specific results
round0_perf = uel._log.permutation_prediction_performance(round_id=0)

# Visually explore the data
uel.explorer()

```

The above examples are indicative, refer above in this document for outline of the full features available after `uel.run` completes operation. 