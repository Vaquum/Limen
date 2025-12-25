# Universal Experiment Loop

Universal Experiment Loop (UEL) is an integral part of Loop, and takes as its input data and a Single File Decoder (SFD). `UEL` currently wraps onto itself (i.e. the object `uel.run` yields) all the folds from `Data` to `Backtest`. In other words, all the following folds are wrapped into one workflow `uel.run`:

- [`Data`](HistoricalData.md)
- [`Indicator`](Indicators.md)
- [`Feature`](Features.md)
- [`SFD`](Single-File-Decoder.md)
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

`Choose Data` -> `Choose Indicators` -> `Choose Features` -> `Develop SFD` -> `Run UEL` -> `Analyze Experiment Log` -> `Analyze Experiment Confusion Metrics` -> `Analyze Backtest Results` -> `Refine Parameters` -> `Run UEL` -> `...`

For an SFD to become mature and ready for trading, one must iterate between running `UEL` and refining parameters many times. Generally speaking, even a relatively small parameter space requires thousands or tens of thousands of permutation rounds before meaningful analytical power is unlocked.

## Refining Parameters

Refining parameters can be understood through expanding or contracting parameters or parameter value ranges. 

## Data

A key point here is that all individual contributors work based on the same underlying data. We achieve this by always calling data from the provided (klines) endpoints available through [HistoricalData](Historical-Data.md). If you don't find what you need through these endpoints, [make an issue](https://github.com/Vaquum/Loop/issues/new) that requests the data that you need, or make a PR that commits the proposed change.

**Declarative manifest approach:** Manifest-based SFDs can configure data sources in their manifest, enabling UEL to automatically fetch data. See [Experiment Manifest](Experiment-Manifest.md) for details.

## SFD

An SFD contains all parameters, data preparation code, and model operation codes in a single python file. For example, representing a logistic regression binary classifier.

**Foundational SFDs** are the official reference SFDs provided by Loop (all use manifest-based configuration). **Custom SFDs** are user-defined SFDs with flexibility to use either manifest-based configuration or custom implementation. Manifest-based SFDs include a `manifest()` function that configures the entire experiment pipeline and enables automatic data fetching.

Read more in [Single File Decoder](Single-File-Decoder.md).

## `UniversalExperimentLoop`

Initializes the Universal Experiment Loop.

### Args

**Note:** All arguments are keyword-only (use `data=...`, `single_file_decoder=...`).

| Parameter             | Type               | Description                                           |
|-----------------------|--------------------|-------------------------------------------------------|
| `data`                | `pl.DataFrame`     | Optional. Experiment data. Required for SFDs using custom functions approach. For manifest-based SFDs, if not provided, data is automatically fetched from configured sources based on `LOOP_ENV` environment variable (defaults to 'test'). |
| `single_file_decoder` | `SingleFileDecoder`| The single file decoder to use for the experiment.    |


### `run`

Runs the experiment `n_permutations` times.

**NOTE:** For **custom SFDs**, you can override any or all of `params`, `prep`, or `model` by passing them as input arguments. For **foundational SFDs**, the manifest auto-generates `prep` from declarative configuration and specifies `model` via `.with_model()`. When overriding, ensure inputs and returns match the contracts outlined in [Single File Decoder](Single-File-Decoder.md).

### Args

| Parameter                     | Type              | Description                                                                                             |
|-------------------------------|-------------------|---------------------------------------------------------------------------------------------------------|
| `experiment_name`             | `str`             | The experiment name. Also used as the CSV filename written in the project root (`<name>.csv`).          |
| `n_permutations`              | `int`             | Number of permutations to run (default: 10000).                                                         |
| `prep_each_round`             | `bool`            | If `True`, calls `sfd.prep(data, round_params=...)` on every round. If `False`, calls `sfd.prep(data)` once on the first round. |
| `random_search`               | `bool`            | If `True`, sample the parameter space randomly; if `False`, iterate deterministically (grid-like).      |
| `maintain_details_in_params`  | `bool`            | If `True`, injects `_experiment_details` (e.g., current index) into `round_params` for logging, then removes it before writing results. |
| `context_params`              | `dict`            | Extra context passed into `round_params` for every round (useful for identifiers like symbol, horizon). |
| `save_to_sqlite`              | `bool`            | If `True`, appends results to SQLite at `/opt/experiments/experiments.sqlite` under table `experiment_name`. |
| `params`                      | `Callable`        | Optional override for the SFD `params` function. Must return a parameter space dictionary.              |
| `prep`                        | `Callable`        | Optional override for the SFD `prep` function. Must follow the standard input/output contract.          |
| `model`                       | `Callable`        | Optional override for the SFD `model` function. Must follow the standard input/output contract.         |

### Returns

Adds artifacts into the `UniversalExperimentLoop` instance and writes streaming logs:

- CSV logging: On the first round, writes a header row to `<experiment_name>.csv`, then appends one result row per round.
- Optional SQLite logging: If `save_to_sqlite=True`, appends the last written row to `/opt/experiments/experiments.sqlite` under table `experiment_name`.

#### Artifacts on the `UniversalExperimentLoop` instance

- `data`: The `pl.DataFrame` used by the loop (explicitly passed or auto-fetched from manifest)
- `params`: The parameter space in use (from SFD `params()` or `params` override in `run()`)
- `prep`: The data preparation function (auto-generated from manifest for manifest-based SFDs, from SFD `prep()` for custom functions approach, or `prep` override in `run()`)
- `model`: The model function (from manifest via `with_model()` for manifest-based SFDs, from SFD `model()` for custom functions approach, or `model` override in `run()`)
- `round_params`: `list[dict]` of the actual parameter sets used in each permutation
- `experiment_log`: `pl.DataFrame` containing accumulated round results (first row created on round 0, then vstacked)
- `extras`: `list[Any]` of any extra artifacts returned by the SFD (when `round_results` contained an `extras` key)
- `models`: `list[Any]` of models returned by the SFD (when `round_results` contained a `models` key)
- `preds`: `list[Any]` predictions captured from the SFD when `round_results['_preds']` is present
- `scalers`: `list[Any]` scalers captured from SFD `prep` when the data dict includes `_scaler`
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


**NOTE**: For fully reproducible post-experiment analysis with `Log`, if using `prep_each_round=True`, make sure that `sfd.prep` is not any random operations, or if you must have random operations, use parametric seeds for round-by-round reproducibility.

### Example

```python
import loop
from loop import sfd

# Manifest-based SFD with auto-fetch
# Data is automatically fetched from manifest-configured sources
uel = loop.UniversalExperimentLoop(single_file_decoder=sfd.foundational_sfd.logreg_binary)

uel.run(
    experiment_name='exp_logreg',
    n_permutations=100,
    prep_each_round=True,
    random_search=True,
)

# Post-run analysis via precomputed artifacts and internal Log
backtest_df = uel.experiment_backtest_results
confusion_df = uel.experiment_confusion_metrics

# Compute round-specific results
round0_perf = uel._log.permutation_prediction_performance(round_id=0)

# Visually explore the data
uel.explorer()

```

The above examples are indicative, refer above in this document for outline of the full features available after `uel.run` completes operation. 