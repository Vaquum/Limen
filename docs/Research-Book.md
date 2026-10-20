# From a research question to a Sensor

A model can fit the past and still fail as a trading signal. The research question is what remains after chronological selection, declared costs and the handoff to inference. Limen lets you own that methodology in Python, or adopt the declared pipeline through Compiled Research.

These projects use recorded Bitcoin observations. Their results describe those observations and decisions; they do not establish future returns. The same research question does not require identical models or make unlike evaluation populations comparable.

## Prerequisites

Use the recorded input bundle and its matching Limen wheel. The project pages state the working directory and optional dependencies. This chapter uses the selection scenario; search, recovery and assessment are separate decisions.

## Choose who owns the experiment

Python Research keeps data, preparation, fitting, backtesting and selection in your functions. Compiled Research records the Manifest definition and applies its configured checks. Invoking that Manifest from Python remains Compiled Research.

Both paths use UEL and MSQ. The difference is the methodological contract you adopt, rather than the language of the driver. Read [the two interfaces](../README.md#two-interfaces-different-purposes) before deciding which responsibility you want Limen to carry.

## Begin with recorded observations

The selection scenarios use `spot_15m_20250101_20250531.parquet`, recorded quarter-hour Bitcoin bars. Its SHA-256 is `eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a`. The Python example reads the first 1,920 observations: 960 for fitting, 480 for validation and 480 reserved for subsequent inference.

Each split loses warm-up and forward-label rows before its partition backtest. Live inference has preceding completed context. Those populations differ; a result should name which one it measures. The source clocks and [data contract](Historical-Data.md) determine when an observation can be used.

## Own the Python methodology

The Python project fits Ridge models to a momentum feature. Every candidate receives the same warm-up exclusion. A decision trades the next bar, with a declared two-basis-point round-trip cost. Search sees fitting and validation observations only. Selection chooses the best validation return per window, using alpha to break ties.

<!-- example: python-selection source=docs/examples/python-research/research.py region=experiment scenario=selection interface=python role=executed -->
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

The selected Sensors wrap the same fitted pipelines that were evaluated. Preprocessing is retained with the model. `Cohort.from_python` averages compatible next-bar return forecasts; it does not convert their units or certify the caller's methodology. See [Python Sensor construction](Trainer.md#python-construction) and [Cohort composition](Cohort.md).

## Adopt the Compiled pipeline

The Compiled project declares a chronological split, a rate-of-change indicator, a fitted quantile target, scaling and logistic regression. The driver resolves its public input-directory argument into the recorded run definition. It runs that definition, selects a validation result and asks Trainer to reconstruct it with the original split.

<!-- example: compiled-selection source=docs/examples/compiled-research/research.py region=experiment scenario=selection interface=compiled role=executed -->
```python
def run_research(data_dir, output_dir):
    config = YAML(typ="safe").load(Path(__file__).with_name("selection.yaml"))
    source = data_dir / "spot_15m_20250101_20250531.parquet"
    expected = "eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Recorded quarter-hour input digest differs")
    # Resolve the driver's public --data-dir argument into the recorded run definition.
    config["sfd"]["manifest"]["data_source"]["params"]["file_path_or_url"] = str(source.resolve())
    compiled = CompiledSFD(config)
    loop = UniversalExperimentLoop(
        sfd=compiled, search_strategy=build_search_strategy(config),
        experiment_dir=output_dir, yaml_reference=config,
    )
    loop.run("compiled-selection", n_permutations=2, workers=1,
             post_processing=True, progress_bar=False)
    selected = completed_log(loop).sort("val_backtest_total_return", descending=True)["id"].to_list()[:1]
    sensors = Trainer(output_dir).train(selected)
    return loop, sensors, selected
```

Reconstruction validates the replayed numeric metrics against the recorded round. It is a refit under the recorded contract; it is not restoration of arbitrary saved weights. [Trainer](Trainer.md#training-and-validation) explains that boundary. A configured pipeline check also does not prove that an arbitrary feature is causal or that source observations are correct.

## Run the complete source

The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) give the entry-point commands and required inputs. Their source is the source executed by documentation tests. A partial chapter excerpt does not replace the complete project or its preparation functions.

## Observe the retained run

These are retained outputs of the identified successful run. Their counts and selected parameters are exact. The Python forecast permits only the numerical tolerance stated in its evidence record. The current validation is shown separately; it does not change when these observations were made.

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

Continue to [search](Advanced-Search.md), [backtesting](Backtest.md) and [inference](Trainer.md) when a decision needs more detail. The book develops those decisions; the references retain the exact names, defaults and failure conditions.
