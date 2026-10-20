# Single-File Decoder

A Single-File Decoder (SFD) is the experiment unit used by both CLI and Python/MSQ runs. It packages a parameter space together with either a declarative manifest or fully custom preparation and model functions.

`limen run` compiles YAML into a manifest-backed SFD. Python callers supply an SFD and search strategy directly. Both routes execute through UEL with MSQ; author a custom SFD when preparation or model code needs arbitrary Python logic.

## Prerequisites

- familiarity with the [Experiment Manifest](Experiment-Manifest.md) and [Universal Experiment Loop](Universal-Experiment-Loop.md)
- a parameter space whose values are lists
- either a concrete manifest or custom `prep()` and `model()` functions

## Choose the SFD style

Limen supports two SFD styles.

| Style | Fit | Required functions | Data handling |
|---|---|---|---|
| manifest-driven | standard Limen experiments, reproducible shared research, built-in workflows | `params()`, `manifest()` | data can be fetched automatically from the manifest |
| custom functions | non-standard prep logic, external libraries, experimental flows | `params()`, `prep()`, `model()` | caller passes `data=` explicitly to `UniversalExperimentLoop` |

Choose by methodological ownership. The [Python project](examples/python-research/README.md) supplies ordinary functions and retains its evaluated fitted models. The [Compiled project](examples/compiled-research/README.md) adopts the Manifest contract and reconstructs selected recorded rounds. Neither requires the other interface as a preliminary step.

## What every SFD must expose

Every SFD must expose `params()`.

The Python project's `run_research` function below supplies a four-candidate `space` through `params=lambda: space`. A normal module-level `params()` can return the same dictionary; the callable boundary is unchanged.

`params()` returns the search space dictionary. Each key is a parameter name and each value must be a list, even when only one value is present.

### `params()` rules

- the return value must be a dictionary
- every value must be a list
- individual values are scalars such as ints, floats, strings, booleans, or callables
- structured values are possible when a manifest step explicitly expects them; keep them deterministic and inspectable

## Manifest-driven SFDs

Manifest-driven SFDs expose `manifest()` instead of custom `prep()` and `model()` functions.

<!-- example: sfd-compiled-definition source=docs/examples/compiled-research/research.py region=experiment scenario=selection interface=compiled role=executed -->
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

This project uses `CompiledSFD` to translate its YAML definition into a Manifest-based SFD. Packaged foundational SFDs expose the same boundary through native `params()` and `manifest()` functions.

### What the manifest path provides

- declarative data fetching
- split-first preparation with train-only fitting
- automatic preparation/model wiring inside UEL
- an inspectable definition for reconstruction and collaboration

### Runtime rules for manifest-driven SFDs

The ordinary MSQ path prepares data for every candidate using the Manifest's preparation and model functions. The search strategy owns the parameter domain; the Manifest owns the pipeline. When `data=` is omitted, Limen fetches data from the Manifest. The [legacy execution reference](Universal-Experiment-Loop.md#legacy-execution-reference) records the older `prep_each_round` and callable-override rules.

## Custom SFDs

Python Research supplies `prep()` and `model()` directly. The complete project constructs its own parameter domain, evaluates caller-owned Ridge pipelines and retains those same fitted objects for inference:

<!-- example: sfd-python-definition source=docs/examples/python-research/research.py region=experiment scenario=selection interface=python role=executed -->
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

The function supplies caller-owned `prep` and `model` callables through a namespace. Its model returns an explicit validation net return and retains the exact fitted pipeline later wrapped in a Sensor. Use the complete project for its imports, feature/target preparation, fitting and input checks.

## Function Contracts

### `prep(data, round_params)`

Custom `prep()` receives the experiment dataframe and the round-specific parameters. It must return a `data_dict` that the model function can consume.

Defaults:

- keep `datetime` in the dataframe until just before `split_data_to_prep_output()`
- pass split lists and column lists directly to `split_data_to_prep_output()`; it copies both inputs before dropping `datetime`, so caller-owned values are unchanged
- capture `all_datetimes = data['datetime'].to_list()` before dropping rows when alignment metadata is required
- make the function deterministic with respect to `round_params`

### `model(data, round_params)`

Custom `model()` receives whatever prepared value the caller's `prep()` returns, plus the round parameters. It returns a results dictionary. The Python project supplies `validation_net_return`; callers may use Limen's binary, multiclass or continuous metrics when those measures fit their model outputs.

Include `_preds` when native prediction recording or downstream analysis needs predictions. Include fitted preprocessing objects under the supported data keys, such as `_scaler`, when using helpers that consume that state. Neither an arbitrary score dictionary nor a retained Python object creates a reconstructible Compiled experiment. The project retains its evaluated scaler inside the same sklearn pipeline as its fitted Ridge model.

## Foundational versus custom

Limen ships foundational SFDs under `limen.sfd.foundational_sfd`. These are reference implementations that show the preferred style for shared Limen workflows.

Custom SFDs are project experiment modules. They can use the same manifest system or the custom `prep()` and `model()` path.

## Foundational SFDs versus reference architecture

There is one more design split that matters in practice:

| Layer | Owns |
|---|---|
| foundational SFD | `params()` plus the packaged manifest |
| reference architecture | the class-based model contract and the function wrapper used by the manifest |

For example:

- `limen.sfd.foundational_sfd.logreg_binary` packages the experiment
- `limen.sfd.reference_architecture.logreg_binary` owns the model implementation

For reconstructible Compiled runs, [Trainer](Trainer.md) replays a selected recorded round with its declared architecture and preprocessing. Python Research instead wraps its retained evaluated fitted state directly.

Execution checks optional dependencies at the model boundary. Importing the SFD catalog does not require every model extra.

Use [Built-In SFDs](Built-In-SFDs.md) for the current shipped catalog and [Reference Architecture](Reference-Architecture.md) for the model layer underneath it.

## Read next

- Continue to [Command-Line Interface](Command-Line-Interface.md) for the normal YAML run path.
- Continue to [Experiment Manifest](Experiment-Manifest.md) for the full declarative pipeline used by standard SFDs.
- Continue to [Built-In SFDs](Built-In-SFDs.md) for the shipped foundational decoder catalog.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for direct Python execution.
- Continue to [Reference Architecture](Reference-Architecture.md) for model implementation authoring beyond packaged decoders.
- Use [Indicators](Indicators.md), [Features](Features.md), [Transforms](Transforms.md), and [Scalers](Scalers.md) as the reference layer while authoring new SFDs.
