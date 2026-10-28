<div align="center">
  <br />
  <a href="https://github.com/Vaquum"><img src="https://github.com/Vaquum/Home/raw/main/assets/Logo.png" alt="Vaquum" width="150" /></a>
  <br />
</div>
<br />
<div align="center">
  <a href="#limen">Limen</a> •
  <a href="#two-interfaces-different-purposes">Two Interfaces</a> •
  <a href="#choose-your-path">Choose Your Path</a> •
  <a href="#python-research">Python Research</a> •
  <a href="#compiled-research">Compiled Research</a> •
  <a href="#shared-safety">Shared Safety</a> •
  <a href="#supply-chain-and-provenance">Package Provenance</a> •
  <a href="#learn-more">Learn More</a>
</div>
<br />
<div align="center">
  <a href="https://www.bestpractices.dev/projects/11898"><img src="https://www.bestpractices.dev/projects/11898/badge" alt="OpenSSF practices badge" /></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/Vaquum/Limen"><img src="https://img.shields.io/ossf-scorecard/github.com/Vaquum/Limen?label=openssf+scorecard&amp;style=flat" alt="OpenSSF Scorecard" /></a>
  <a href="https://pypi.org/project/vaquum-limen/"><img src="https://img.shields.io/pypi/v/vaquum-limen?label=pypi" alt="PyPI version" /></a>
  <a href="https://docs.vaquum.fi/limen/"><img src="https://img.shields.io/badge/docs-limen-blue" alt="Limen docs" /></a>
  <a href="https://github.com/Vaquum/Limen/actions/workflows/pr_checks_tests.yml"><img src="https://github.com/Vaquum/Limen/actions/workflows/pr_checks_tests.yml/badge.svg" alt="PR tests" /></a>
</div>

<hr />

<a id="limen"></a>

# The Strategy Tested is the Strategy Deployed

Limen is a research engine for Bitcoin traders that brings together parameter sweep for machine learning and rule-based models, financially rigorous backtesting, and trading sensor cohort selection.

Limen helps you go from hypothesis to a trading-ready sensor or sensor cohort with freedom and speed that set a new standard for similar systems.

## Two interfaces, different purposes

Limen offers two interfaces to one research engine: **Python Research** and **Compiled Research**. Python Research gives you unrestricted control: bring your own data, models, backtesting and selection logic, using ordinary Python without a new language or programming model. Compiled Research turns strictly typed manifests into experiments through the CLI, enforcing safeguards against common financial-modelling pitfalls and producing inspectable evidence of what was checked. Both share the same execution safety; one gives you complete methodological freedom, while the other makes its methodological constraints explicit and enforceable.

## Choose your path

| Your choice | Python Research | Compiled Research |
|---|---|---|
| Choose it when | You want to own the research methodology and every component. | You want Limen to enforce the declared experiment's research constraints. |
| Define the experiment in | Ordinary Python, with your choice of libraries and functions. | A strictly typed YAML manifest, compiled and run through the CLI. |
| Control data and models | Bring your own sources, preparation, models, backtesting and selection. | Declare sources, splits, features, targets, models and execution assumptions within the Manifest contract. |
| Checks and evidence | You implement the preparation, evaluation and evidence you need. | Limen applies the configured pipeline safeguards and records the run definition and configured evidence. |
| Start here | [Start Python Research](#python-research) | [Start Compiled Research](#compiled-research) |

## Install

Both paths use the same installation:

<!-- example: readme-install source=docs/examples/commands.json region=install scenario=selection interface=python role=executed -->
```bash
python -m pip install "vaquum-limen[data]"
```

Requires Python `>=3.10,<3.14` (3.10–3.13) on macOS or Linux. Optional extras: `boosting` for LightGBM/XGBoost, `tabpfn` for TabPFN, `neural` for LSTM, `hmm` for GaussianHMM, `indicators` for TA-Lib comparison tooling, `stats` for statistical helpers, or `vaquum-limen[all]` for the full stack.

## Python Research

Bring your own data and define the experiment in ordinary Python. Your functions own the research methodology, including features, targets, backtesting strategies and cohort selection.

The path is **your data → experiment → search → selection → fitted models → Sensor**. The [complete Python project](https://github.com/Vaquum/Limen/blob/main/docs/examples/python-research/README.md) defines every research component explicitly. Replace any function with your own Python or compose it with Limen's built-in tools. These excerpts belong to that complete source, including its imports, entry point and recorded input checks.

### Define your experiment

Start with a momentum feature and the next bar's open-to-close return:

<!-- example: readme-python-feature source=docs/examples/python-research/research.py region=momentum_feature scenario=selection interface=python role=executed -->
```python
def features(bars, window):
    return bars.with_columns(
        (pl.col("close") / pl.col("close").shift(window) - 1).alias("momentum")
    )
```

<!-- example: readme-python-target source=docs/examples/python-research/research.py region=next_return_target scenario=selection interface=python role=executed -->
```python
def prepare_split(bars, round_params):
    return features(bars, round_params["window"]).with_columns(
        (pl.col("close").shift(-1) / pl.col("open").shift(-1) - 1)
        .alias("next_return")
    ).slice(12).drop_nulls(["momentum", "next_return"])
```

Split before computing features and forward labels. Each candidate loses the same 12-bar warm-up and its partition's final forward-label row:

<!-- example: readme-python-split source=docs/examples/python-research/research.py region=split_preparation scenario=selection interface=python role=executed -->
```python
def prep(bars, round_params):
    # Split before features and labels; no target crosses a boundary.
    return {
        "train": prepare_split(bars[:960], round_params),
        "validation": prepare_split(bars[960:1440], round_params),
    }
```

Fit a scaler and Ridge model on training observations only, then score the validation forecasts with an explicit one-bar policy:

<!-- example: readme-python-fit source=docs/examples/python-research/research.py region=fitted_scaler scenario=selection interface=python role=executed -->
```python
def fit(train, round_params):
    return make_pipeline(StandardScaler(), Ridge(alpha=round_params["alpha"])).fit(
        train.select("momentum"), train["next_return"].to_numpy()
    )
```

<!-- example: readme-python-score source=docs/examples/python-research/research.py region=validation_score scenario=selection interface=python role=executed -->
```python
def backtest(validation, forecasts):
    positions = np.sign(forecasts)
    net = positions * validation["next_return"].to_numpy() - abs(positions) * 0.0002
    return float(np.mean(net))
```

The score uses signed forecasts, fixed notional and a two-basis-point round-trip cost. This calculation belongs to the caller; it is separate from Limen's configured event execution policy.

### Run the experiment

The project searches four candidates, retains their evaluated fitted pipelines and chooses the best validation result per feature window. Alpha breaks ties:

<!-- example: readme-python-experiment source=docs/examples/python-research/research.py region=experiment scenario=selection interface=python role=executed -->
```python
def run_research(bars, experiment_name):
    research = bars[:1440]
    uel, retained = run_trials(research, experiment_name)
    selected = choose_parameters(uel)
    sensors = selected_sensors(bars, selected, retained)
    cohort = Cohort.from_python(sensors)
    return cohort, uel, selected, retained
```

Use the project and recorded input bundle linked with the example evidence. From the bundle root, run:

<!-- example: readme-python-command source=docs/examples/commands.json region=python-selection scenario=selection interface=python role=executed -->
```bash
python python-research/research.py --scenario selection --data-dir inputs --output-dir results/python
```

The command checks the recorded source digest and writes `research.csv` and `summary.json`. The evaluated models and Cohort remain in caller-owned memory during the run. Import `run_research` to continue using the returned objects in the same process; the two output files do not restore arbitrary fitted state.

### Create a Sensor

The function above wraps each selected evaluated pipeline with `Sensor.from_model`. The Sensor reuses the fitted scaler and model without refitting, selects the feature names recorded during fitting, and blocks null or nonfinite features and decisions before the selection cutoff. Pass completed history to `sensor.predict(history)` for its latest numeric forecast or explicit unavailable reason. Availability remains a separate declared clock.

Your Python still owns preparation and causality. Custom prediction and output mapping remain available, while `Sensor.from_python` accepts arbitrary fitted state and inference logic. See [Python Sensor construction](https://github.com/Vaquum/Limen/blob/main/docs/Trainer.md#python-construction).

### Create a sensor cohort

Your own Python selects any number of candidates from the results. The project combines its two selected Sensors with `Cohort.from_python`, averaging compatible next-bar return forecasts. Call `cohort.predict(history)` just like a Sensor. Members declare the same `output_id` to identify compatible meanings; Limen checks timestamp alignment and preserves unavailable results. Binary voting and custom aggregation are also available. See [Python cohorts](https://github.com/Vaquum/Limen/blob/main/docs/Cohort.md#python-cohorts).

This recorded case fits on the first 960 observations, compares candidates on the next 480 and reserves the final 480 for later inference and assessment. The project continues through search, feedback, recovery, walk-forward, calibration, execution and holdout decisions. Replace any research component with your own Python; the one-feature model and one-bar score are the starting point.

Continue with [custom SFDs](https://github.com/Vaquum/Limen/blob/main/docs/Single-File-Decoder.md#custom-sfds), [UEL](https://github.com/Vaquum/Limen/blob/main/docs/Universal-Experiment-Loop.md#direct-python-run), or [search and feedback](https://github.com/Vaquum/Limen/blob/main/docs/Advanced-Search.md).

## Compiled Research

Declare the experiment and let Limen enforce its pipeline contract. The Manifest specifies the data, chronological splits, preparation, model and parameter space; the CLI validates and compiles that definition before executing it. The YAML is the complete research definition; run it through the CLI.

The [complete Compiled project](https://github.com/Vaquum/Limen/blob/main/docs/examples/compiled-research/README.md) starts with a recorded-data logistic experiment. Its selection declaration fixes the source, dates, train-fitted target/scaler, execution policy and validation objective:

<details>
<summary>Read the full YAML manifest</summary>

<!-- example: readme-compiled-definition source=docs/examples/compiled-research/selection.yaml region=__file__ scenario=selection interface=compiled role=executed -->
```yaml
# region declaration
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
# endregion declaration
```

</details>

From the example bundle root, run that exact declaration through the installed CLI:

<!-- example: readme-compiled-command source=docs/examples/commands.json region=compiled-yaml scenario=selection interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/selection.yaml
```

The command validates and compiles the declaration, executes two candidates and writes `results/dev/compiled-cli`. The [CLI lifecycle](https://github.com/Vaquum/Limen/blob/main/docs/Command-Line-Interface.md#first-yaml-run) also exercises template initialization, explicit validation/profiling, dry compilation, immutable committed definitions, recovery and project backup.

<!-- example: readme-compiled-assessment source=docs/examples/commands.json region=compiled-selection-assess scenario=selection interface=compiled role=executed -->
```bash
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
```

After execution, `limen assess` applies the YAML's validation selection, reconstructs the selected Sensors and binds their Cohort. Reconstruction refits under the recorded pipeline and checks replayed metrics. The [comprehensive YAML](https://github.com/Vaquum/Limen/blob/main/docs/examples/compiled-research/research.yaml) continues through feature ablation, PCA, calibration, threshold selection, full recording and inference; separate declarations cover incompatible research choices.

| Artifact | What it records |
|---|---|
| Copied YAML manifest | The experiment definition. |
| `metadata.json` | Run configuration and provenance metadata. |
| `results.csv` | Candidate results. |
| `round_data.jsonl.gz` | Round identities, parameters and payloads selected by the recording policy. |

Row evidence and recovery depend on the [recording policy](https://github.com/Vaquum/Limen/blob/main/docs/Universal-Experiment-Loop.md#recording-policy). Choose it before a run when later replay or continuation needs those payloads.

Continue with the [CLI workflow](https://github.com/Vaquum/Limen/blob/main/docs/Command-Line-Interface.md), [Manifest contract](https://github.com/Vaquum/Limen/blob/main/docs/Experiment-Manifest.md), or [end-to-end workflow](https://github.com/Vaquum/Limen/blob/main/docs/End-to-End-Workflow.md).

## Shared safety

Both interfaces share the search engine and reusable research components. Python Research lets you compose or replace them; Compiled Research wires their safeguards into a checked pipeline.

### Common foundations

MSQ and UEL provide parameter-domain checks, parameter identities, explicit budgets, round results and convergence-warning records. Seeded built-in search supports repeatable candidate ordering. Optional persistence adds checkpoint and artifact-integrity checks; recovery requires a compatible checkpoint and full recording. Search seeds do not freeze models, data or environments. See [search and recovery](https://github.com/Vaquum/Limen/blob/main/docs/Advanced-Search.md).

In both interfaces, native Sensors wrap fitted state and distinguish usable predictions from unavailable decisions. Native Cohorts check member timestamp alignment before aggregation. Python callers supply their own preparation, selection cutoff and output meanings; these declarations do not create compiled provenance or methodological proof.

In Python Research, component safeguards apply to the Limen components you use; custom replacements define their own behavior.

### What Compiled Research adds

The CLI validates schema, component references, parameter links, split declarations and execution settings before running. It coordinates the following controls for the declared pipeline:

| Common pitfall | Pipeline protection |
|---|---|
| Features or forward labels crossing evaluation boundaries | Split raw observations before bar formation, indicators, features and target construction. Forward targets are constructed within each partition. |
| Fitting preprocessing on future data, or feeding labels into the model | Fit target thresholds, scalers, PCA and fitted feature transformations on training data. Reuse that state elsewhere; exclude the target and its declared source columns from model features. |
| Calibrating probabilities or decision thresholds on the test set | Fit configured calibration and thresholds on validation, then reuse them on test data and at inference. |
| Warm-up effects, missing features and misaligned prices | Carry historical context; align retained features with original prices by timestamp, then score predictions in that row order. Packaged ML templates enable strict null checking, marking affected rounds as failed. |
| Invalid rule conditions or history windows | Validate condition identities, dependencies and temporal parameters; evaluate trailing persistence and recency within each split. |
| Inflated returns from omitted trading assumptions | Apply execution lag, fees, slippage and notional settings through the selected backtester. Configured stops verify OHLC provenance and use explicit gap and same-bar exit rules. |
| Impossible fills, unavailable prices or incorrect funding charges | Configured event execution checks availability clocks, source coverage, exposure, quantity and collateral rules. Funding adapters validate units, schedules and valuation evidence and separate accrual from settlement. |
| Research and inference quietly using different pipelines | Trainer reconstructs the declared pipeline and checks replayed results and recorded bindings where available. Compiled Sensors carry manifest and permutation identities and reuse fitted preprocessing; artifact-backed Cohorts bind the selected member identities. |

See the [preparation contract](https://github.com/Vaquum/Limen/blob/main/docs/Experiment-Manifest.md#how-a-manifest-executes), [backtest mechanics](https://github.com/Vaquum/Limen/blob/main/docs/Backtest.md) and [inference contract](https://github.com/Vaquum/Limen/blob/main/docs/Trainer.md#sensor) for each control's scope.

### Additional validation, when enabled

- [Walk-forward validation](https://github.com/Vaquum/Limen/blob/main/docs/Experiment-Manifest.md#walk-forward-sweeps) refits each fold with declared purge and embargo gaps. Choose gaps that cover your label horizon.
- A [validation-return objective](https://github.com/Vaquum/Limen/blob/main/docs/Calibration.md#shared-validation-return-objective) binds supported threshold tuning and reducers to validation execution. [DSR and PBO reports](https://github.com/Vaquum/Limen/blob/main/docs/Benchmark.md#walk-forward-selection-report) assess retained research evidence; failed or unavailable criteria are reported rather than stopping the run.
- Opt-in [project holdout reservations](https://github.com/Vaquum/Limen/blob/main/docs/Final-Holdout.md) exclude reserved observations from supported research and replay. Active reservations reject custom SFDs and unsupported modes; see the workflow's restrictions.

Compilation does not establish arbitrary feature causality, data correctness or future performance. Validation used for tuning is selection evidence; statistical reports do not create an untouched holdout.

## Research scope

Limen handles research, backtesting and the creation of reusable Sensors and Cohorts. Its built-in components focus on Bitcoin models and strategies. The surrounding Vaquum tools have distinct roles:

| Tool | Role |
|---|---|
| [Origo](https://github.com/Vaquum/Origo) | Ingests Bitcoin market data, forms bars and publishes datasets for research and live consumers. |
| [Grid](https://github.com/Vaquum/Grid) | Provides the visual workspace for authoring and running Limen manifests, monitoring sweeps, analysing parameter effects and uncertainty, and building sensor cohorts from recorded results. |
| [Nexus](https://github.com/Vaquum/Nexus) | Turns prediction signals into trading decisions, applying risk checks and capital allocation. |
| [Praxis](https://github.com/Vaquum/Praxis) | Submits and manages venue orders, tracks fills and positions, and records trade outcomes. |

You can also integrate Limen's Sensors and Cohorts into your own trading system.

## Risk Boundary

Limen is research software. Benchmark and backtest outputs are not investment advice, trading advice, execution simulation, regulatory approval, or a promise of future performance. Past performance is not predictive, and trading digital assets can result in total loss of capital.

## Supply chain and provenance

Limen makes its package origin, build process and release evidence inspectable:

- **Reviewed source.** Protected merges require approval and passing checks for tests, coverage, static analysis, dependency vulnerabilities and packaging.
- **Controlled builds.** Workflow actions are pinned to full commit hashes; CI requirements are installed with hash verification. Packaging checks build the wheel and source distribution twice in the same CI environment, with a fixed timestamp, and compare their SHA-256 digests.
- **Dependency maintenance.** Weekly dependency update proposals accompany vulnerability audits of declared core dependencies. Exceptions require a recorded reason and expiry date.
- **Verifiable releases.** PyPI publishing authenticates through OpenID Connect using Trusted Publishing. GitHub releases carry the same wheel and source distribution, build-provenance attestations, SHA-256 digests and a CycloneDX software bill of materials (SBOM) describing the release environment.

Limen holds [OpenSSF Best Practices Silver](https://www.bestpractices.dev/en/projects/11898/silver), based on a public project self-assessment. Silver's [requirements](https://www.bestpractices.dev/en/criteria/1) include repeatable builds, cryptographically signed releases, dependency inventory and monitoring, automated testing and documented security assurance.

See how to [verify a release](https://github.com/Vaquum/Limen/blob/main/docs/Developer/Release-Policy.md#verifying-release-artifacts), inspect the [security assurance case](https://github.com/Vaquum/Limen/blob/main/docs/Developer/Security-Assurance-Case.md), or [report a vulnerability](https://github.com/Vaquum/Limen/blob/main/SECURITY.md).

## Learn more

| Path | Guides |
|---|---|
| Python Research | [Search and feedback](https://github.com/Vaquum/Limen/blob/main/docs/Advanced-Search.md), [custom SFDs](https://github.com/Vaquum/Limen/blob/main/docs/Single-File-Decoder.md#custom-sfds), [UEL](https://github.com/Vaquum/Limen/blob/main/docs/Universal-Experiment-Loop.md). |
| Compiled Research | [CLI](https://github.com/Vaquum/Limen/blob/main/docs/Command-Line-Interface.md), [Manifest](https://github.com/Vaquum/Limen/blob/main/docs/Experiment-Manifest.md), [end-to-end workflow](https://github.com/Vaquum/Limen/blob/main/docs/End-to-End-Workflow.md). |
| Research tools | [Features](https://github.com/Vaquum/Limen/blob/main/docs/Features.md), [targets](https://github.com/Vaquum/Limen/blob/main/docs/Targets.md), [backtesting](https://github.com/Vaquum/Limen/blob/main/docs/Backtest.md), [analysis](https://github.com/Vaquum/Limen/blob/main/docs/Log.md). |
| Model reuse | [Trainer](https://github.com/Vaquum/Limen/blob/main/docs/Trainer.md), [Cohort](https://github.com/Vaquum/Limen/blob/main/docs/Cohort.md). |

See the [documentation hub](https://github.com/Vaquum/Limen/blob/main/docs/README.md), [Roadmap](https://github.com/Vaquum/Limen/blob/main/docs/Roadmap.md), and [Security Assurance Case](https://github.com/Vaquum/Limen/blob/main/docs/Developer/Security-Assurance-Case.md).

## Contributing

Contribution starts through [CONTRIBUTING.md](https://github.com/Vaquum/Limen/blob/main/CONTRIBUTING.md), [docs changes](https://github.com/Vaquum/Limen/tree/main/docs), or [open issues](https://github.com/Vaquum/Limen/issues).

Before contributing, start with the [Developer docs](https://github.com/Vaquum/Limen/blob/main/docs/Developer/README.md).

## Support

Use [SUPPORT.md](https://github.com/Vaquum/Limen/blob/main/SUPPORT.md) for support routes and scope boundaries.

## Vulnerabilities

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/Vaquum/Limen/security/advisories/new). Do not report vulnerabilities through public issues.

## Citations

Published work should cite:

Vaquum Limen [Computer software]. (2026). Retrieved from [GitHub](https://github.com/Vaquum/Limen).

Machine-readable citation metadata lives in [CITATION.cff](https://github.com/Vaquum/Limen/blob/main/CITATION.cff).

## License

[MIT License](https://github.com/Vaquum/Limen/blob/main/LICENSE).
