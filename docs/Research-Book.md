# From a research question to a Sensor

A model can fit the past and still fail as a trading signal. The research question is what remains after chronological selection, declared costs and the handoff to inference. Limen lets you own that methodology in Python, or adopt the declared pipeline through Compiled Research.

The projects presented in the following pages use recorded Bitcoin observations available through Limen\. Their results describe those observations and decisions; they do not establish future returns\.

## Prerequisites

Use Python 3\.10–3\.13 and the matching wheel and recorded project bundle\. The acquisition steps below establish the environment and working directory for either Python or Compiled research interface provided by Limen\.

## Acquire the project

Choose the [Python project](<examples/python-research/README.md#acquire-the-matching-project>) or [Compiled project](<examples/compiled-research/README.md#acquire-the-matching-project>)\. Each begins with the same digest\-verified bundle and wheel acquisition, environment and working directory\. You can follow either path independently\. This chapter uses a small selection run; later chapters develop its decisions through named variants\.

<a id="choose-who-owns-the-experiment" data-reader-heading="plain"></a>

## Choose the research interface

Python Research keeps data, preparation, fitting, backtesting and selection in your functions. Compiled Research records the Manifest definition and applies its configured checks. Compiled Research is authored in YAML and executed through the CLI.

Both paths use the same machinery internally\. Python Research owns ordinary Python functions; Compiled Research owns YAML files and CLI commands\. Read [the two interfaces](<../README.md#two-interfaces-different-purposes>) before deciding which responsibility you want Limen to carry on your behalf and the degree of freedom you want to enjoy\.

## Begin with recorded observations

The selection scenarios use `spot_15m_20250101_20250531.parquet`, recorded quarter-hour Bitcoin bars. Its SHA-256 is `eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a`. The Python example reads the first 1,920 observations: 960 for fitting, 480 for validation and 480 reserved for subsequent inference.

Each split loses warm-up and forward-label rows before its partition backtest. Live inference has preceding completed context. Those populations differ; a result should name which one it measures. The source clocks and [data contract](Historical-Data.md) determine when an observation can be used.

## Own the Python methodology

The Python project fits Ridge models to a momentum feature. Every candidate receives the same warm-up exclusion. A decision trades the next bar, with a declared two-basis-point round-trip cost. Search sees fitting and validation observations only. Selection chooses the best validation return per window, using alpha to break ties.

<!-- example: python-selection source=docs/examples/python-research/research.py region=experiment scenario=selection interface=python role=executed -->
```python
def run_research(bars, experiment_name):
    research = bars[:1440]
    uel, retained = run_trials(research, experiment_name)
    selected = choose_parameters(uel)
    sensors = selected_sensors(bars, selected, retained)
    cohort = Cohort.from_python(sensors)
    return cohort, uel, selected, retained
```

The selected Sensors wrap the same fitted pipelines that were evaluated. Preprocessing is retained with the model. `Cohort.from_python` averages compatible next-bar return forecasts; it does not convert their units or certify the caller's methodology. See [Python Sensor construction](Trainer.md#python-construction) and [Cohort composition](Cohort.md).

## Adopt the Compiled pipeline

The Compiled project declares its source, dates, preparation, model, search and selection in YAML. Execute and assess the introductory declaration from the bundle root:

<!-- example: book-compiled-first-command source=docs/examples/compiled-research/run.sh region=selection scenario=selection interface=compiled role=executed -->
```bash
limen validate compiled-research/selection.yaml
limen profile compiled-research/selection.yaml
limen run --dry-run compiled-research/selection.yaml
limen run --no-progress-bar compiled-research/selection.yaml
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
```

Its selection block fixes the validation ranking before assessment. `limen assess` reconstructs the selected Sensors and writes Cohort predictions; the [complete project](examples/compiled-research/README.md) explains their artifacts.

<!-- example: compiled-selection source=docs/examples/compiled-research/selection.yaml region=assessment scenario=selection interface=compiled role=executed -->
```yaml
assessment:
  metric: val_backtest_total_return
  direction: maximize
  top_k: 1
  predict: true
```

Reconstruction validates the replayed numeric metrics against the recorded round. It is a refit under the recorded contract; it is not restoration of arbitrary saved weights. [Trainer](Trainer.md#training-and-validation) explains that boundary. A configured pipeline check also does not prove that an arbitrary feature is causal or that source observations are correct.

## Run the complete source

The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) give the entry-point commands and required inputs. Their source is the source executed by documentation tests. A partial chapter excerpt does not replace the complete project or its preparation functions.

## Observe the retained run

These are retained outputs of the identified successful run. The Python project and unchanged Compiled YAML/CLI commands ran against the same isolated installed wheel. Their counts and selected parameters are exact. The Python forecast permits only the numerical tolerance stated in its evidence record. The current validation is shown separately; it does not change when these observations were made.

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

## Inspect the result before continuing

A result needs more than a score. Inspect its selected parameters, fitting and selection cutoffs, traded observations, costs and inference population. Python proves those choices through caller-owned assertions. Compiled Research supplies the evidence admitted by its declared pipeline and recording policy.

Continue to [recorded observations](Historical-Data.md) to establish the data and clocks the experiment will consume. The Reference appendix retains the exact names, defaults and failure conditions.
