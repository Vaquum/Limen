# From a research question to a Sensor

Limen is a research engine for Bitcoin trading. The engine evaluates model and strategy parameters, records results, and carries selected models into inference. Python Research defines the methodology through ordinary Python functions. Compiled Research declares experiments in YAML and executes them through the configured Limen pipeline.

Research proceeds from **observations → experiment → search → selection → Sensor or Cohort**. A Sensor combines a fitted model, its input preparation, and timing information for inference. A Cohort combines compatible Sensors. The Python and Compiled projects provide executable examples for each stage.

## The interfaces assign research responsibility

The interfaces share one engine and assign different parts of the methodology to the caller.

| Research responsibility | Python Research | Compiled Research |
|---|---|---|
| Experiment definition | Ordinary Python functions for preparation, fitting, evaluation, and selection. | A YAML definition of sources, splits, preparation, models, search, and assessment, executed through the command-line interface (CLI). |
| Engine behavior | Experiment execution, parameter search, and result logging, with research components for use alongside caller-defined code. | Execution of the declared pipeline, its configured safeguards, and recorded evidence for inspection and reconstruction. |
| Caller responsibility | The methodology and its checks, including the interpretation of data, scores, and predictions. | The declaration, its assumptions, and the interpretation of the resulting evidence. |
| Inference handoff | Evaluated fitted state wrapped in Sensors and composed in Python. | Selected recorded rounds reconstructed through `limen assess`, with compatible Sensors bound into a Cohort. |

Python Research supports caller-defined data sources, models, backtesting, and selection in ordinary Python. Compiled Research defines experiments in YAML and executes them through the CLI under the [Manifest contract](Experiment-Manifest.md). Limen’s indicators, features, targets, scalers, and backtesting tools are available through both interfaces.

## Both projects investigate momentum after costs

**The research question is whether recent price momentum supports a trading decision after costs.** Recorded quarter-hour Bitcoin bars supply observations for candidate evaluation, chronological selection, and later inference.

The Python project predicts the return on the next bar with Ridge regression. The Compiled project uses logistic regression with a fitted binary target. The models, date partitions, and trading policies differ. Their scores measure separate experiments.

Both projects require Python 3.10–3.13 and the [matching project bundle and wheel](examples/python-research/README.md#acquire-the-matching-project). The bundle contains the recorded inputs and both complete projects, distributed separately from the installed package. The commands execute from the acquired bundle root with its environment activated. Each excerpt comes from the corresponding complete project.

## Python Research executes caller-defined functions

The Python project splits observations before preparing momentum features and forward labels. A scaler and Ridge model fit the training observations. The backtest trades the next bar using the sign of the forecast and deducts a round-trip cost of 2 basis points. Ordinary Python functions define each step.

The Universal Experiment Loop (UEL) executes the research definition through three functions. `params()` supplies candidate values, `prep()` prepares the data, and `model()` fits and evaluates each candidate. The project functions `fit`, `prep`, and `backtest` define the methodology. Limen evaluates four combinations of momentum window and Ridge alpha, then logs each validation score.

<!-- example: book-python-trials source=docs/examples/python-research/research.py region=trials scenario=selection interface=python role=executed -->
```python
def run_trials(research, experiment_name):
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
    return uel, retained
```

`SimpleNamespace` groups the three functions into a single-file decoder (SFD). `GridStrategy` generates parameter combinations, and the Mutable Search Queue (MSQ) manages the candidates. The functions and search policy remain caller-controlled.

Selection ranks candidates by validation return within each momentum window, using alpha to break ties. `Sensor.from_model` wraps each selected fitted pipeline:

<!-- example: book-python-sensors source=docs/examples/python-research/research.py region=selected_sensors scenario=selection interface=python role=executed -->
```python
def selected_sensors(bars, selected, retained):
    return [
        Sensor.from_model(retained[(p["window"], p["alpha"])], prepare=prepare_live,
                          params=p, fitting_cutoff=bars["datetime"][960],
                          selection_cutoff=bars["datetime"][1440], output_id="next_bar_return")
        for p in selected
    ]
```

Each Sensor retains the evaluated scaler and model. The caller-defined `prepare_live` function computes the matching feature and its availability time. `Cohort.from_python` averages the compatible next-bar return forecasts. The caller defines the prediction meaning and units.

The [complete Python project](examples/python-research/README.md#run-the-selection-scenario) executes with this command:

<!-- example: book-python-command source=docs/examples/commands.json region=python-selection scenario=selection interface=python role=executed -->
```bash
python python-research/research.py --scenario selection --data-dir inputs --output-dir results/python
```

The command writes `research.csv` and `summary.json`. The evaluated models and Cohort remain in process memory. The project function `run_research` returns those objects for continued inference within that process.

## Compiled Research executes a YAML definition

The YAML definition specifies the research choices in Compiled Research. `selection.yaml` names the recorded source and chronological splits. The declared pipeline calculates a one-bar rate of change, fits a binary target and scaler, and evaluates two logistic-regression candidates. Cash-spot execution applies a fee of 1 basis point per fill. Assessment selects the highest validation backtest return.

The declaration specifies data, preparation, fitting, search, and assessment. Changes to supported choices remain within YAML and execute through the CLI.

<details>
<summary>Read the complete selection.yaml</summary>

<!-- example: compiled-selection source=docs/examples/compiled-research/selection.yaml region=declaration scenario=selection interface=compiled role=executed -->
```yaml
schema_version: "1.0"
metadata:
  name: book_compiled_selection
  mode: development
sfd:
  manifest:
    type: ml
# region recorded_source
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_15m_20250101_20250531.parquet
# endregion recorded_source
    split_dates:
      train_start: "2025-01-01"
      train_end: "2025-01-15"
      val_start: "2025-01-15"
      val_end: "2025-01-18"
      test_start: "2025-01-18"
      test_end: "2025-01-22"
# region momentum_indicator
    indicators:
      - func: limen.indicators.roc
        params:
          period: 1
          group: momentum
# endregion momentum_indicator
# region fitted_target
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: roc_1
        quantile: 0.5
      transform_params:
        shift: -1
# endregion fitted_target
# region validation_objective
    objective:
      metric: backtest_total_return
      direction: maximize
# endregion validation_objective
    backtest:
      source_interval_seconds: 900
      fee_bps: 1.0
      slip_bps: 0.0
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1.0e-9
        min_notional: 0.0
# region fitted_scaler
    scaler:
      from_params: scaler_type
# endregion fitted_scaler
    reference_architecture: limen.sfd.reference_architecture.logreg_binary
  params:
    scaler_type: [logreg]
    C: [1.0, 0.5]
    random_state: [42]
uel:
  output_path: compiled-cli
  n_permutations: 2
  search_strategy:
    type: grid
  output_format: csv

# region assessment
assessment:
  metric: val_backtest_total_return
  direction: maximize
  top_k: 1
  predict: true
# endregion assessment
```

</details>

The CLI validates, profiles, executes, and assesses the declaration from the bundle root:

<!-- example: book-compiled-first-command source=docs/examples/compiled-research/run.sh region=selection scenario=selection interface=compiled role=executed -->
```bash
limen validate compiled-research/selection.yaml
limen profile compiled-research/selection.yaml
limen run --dry-run compiled-research/selection.yaml
limen run --no-progress-bar compiled-research/selection.yaml
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
```

`validate` checks the declaration. `profile` reports the search size and estimates runtime by loading data and sampling preparation and fitting. The dry run checks readiness. `run` evaluates candidates and records results. `assess` applies the declared validation ranking, reconstructs the selected Sensor, and writes Cohort predictions. The [complete Compiled project](examples/compiled-research/README.md) documents the artifacts.

Reconstruction refits the selected model under the recorded contract and checks numeric replay metrics against the recorded round. Pipeline checks establish compliance with the configured safeguards. Source accuracy and feature causality require independent verification. Source selection and research assumptions remain caller-controlled.

## Retained results record selection and inference

The retained outputs record candidate evaluation, selection, and inference on the recorded observations:

| Python Research | Compiled Research |
|---|---|
| Evaluated four candidates and retained all four fitted pipelines. Selected `alpha=0.1` for each of the 4-bar and 12-bar momentum windows. | Evaluated two candidates, selected one recorded round and reconstructed one Sensor. |
| Produced 480 later Cohort predictions. The last forecast was approximately 3.68 basis points of next-bar return. | Reconstruction validated numeric replay metrics against the recorded round. |

These results establish execution, selection, and inference on recorded observations. Assessment of the trading question requires validation scores, costs, traded observations, and the later assessment population.

<details>
<summary>Inspect the retained outputs and their execution evidence</summary>

Each output retains its original run identity. Source links identify the recorded inputs, package, and execution evidence. Current validation and retained observations have separate execution dates.

### Python selection result

<!-- example-output: python-selection source=docs/examples/retained/python-selection.json scenario=selection interface=python role=observed -->
```json
{
  "interface": "python",
  "scenario": "selection",
  "rounds": 4,
  "selected": [
    {
      "window": 4,
      "alpha": 0.1
    },
    {
      "window": 12,
      "alpha": 0.1
    }
  ],
  "retained_models": 4,
  "inference_rows": 480,
  "last_prediction": 0.0003680179954790141,
  "fitting_cutoff": "2025-01-11T00:00:00+00:00",
  "selection_cutoff": "2025-01-16T00:00:00+00:00"
}
```

### Compiled selection result

<!-- example-output: compiled-selection source=docs/examples/retained/compiled-selection.json scenario=selection interface=compiled role=observed -->
```json
{
  "interface": "compiled",
  "scenario": "selection",
  "rounds": 2,
  "selected_rounds": 1,
  "reconstructed_sensors": 1
}
```

</details>

## Source clocks govern the next research decision

The selection projects use `spot_15m_20250101_20250531.parquet`. The Python project uses 960 observations for fitting, 480 for validation, and 480 for subsequent inference. Preparation removes warm-up and forward-label rows from each research partition. Inference uses preceding completed context. Result interpretation depends on the population evaluated. The Compiled declaration specifies separate date boundaries.

[Historical Data](Historical-Data.md) establishes the source, its digest, and its availability clocks for both interfaces. Protected final observations require a reserved boundary before research begins.
