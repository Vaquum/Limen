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

See how Limen [compares with other similar systems](https://docs.vaquum.fi/limen/overview/comparison).

<!-- TODO: Create the comparison page covering performance per compute budget, academic modelling rigor throughout the workflow, and safety properties. -->

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

```bash
pip install "vaquum-limen[data]"
```

Requires Python `>=3.10,<3.14` (3.10–3.13) on macOS or Linux. Optional extras: `boosting` for LightGBM/XGBoost, `tabpfn` for TabPFN, `indicators` for TA-Lib comparison tooling, `stats` for statistical helpers, or `vaquum-limen[all]` for the full stack.

## Python Research

Bring your own data and define the experiment in ordinary Python. Your functions own the research methodology, including features, targets, backtesting strategies and cohort selection.

The path is **your data → experiment → search → selection → fitted models → Sensor**. This example defines every research component explicitly. Replace any function with your own Python or compose it with Limen's built-in tools.

### Define your experiment

Save the experiment as `research.py`:

```python
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


fitted_models = {}  # Keep the models evaluated during this serial sweep.


def params():
    return {"window": [12, 24, 48], "C": [0.1, 1.0]}


def features(bars, round_params):
    return bars.with_columns(
        pl.col("close").pct_change(round_params["window"]).alias("momentum")
    )


def prep(bars, round_params):
    # Split first: forward labels never cross a split boundary.
    a, b = int(len(bars) * 0.6), int(len(bars) * 0.8)
    splits = {"train": bars[:a], "validation": bars[a:b], "test": bars[b:]}
    warmup = max(params()["window"])  # Compare the same rows for every candidate.
    return {
        name: features(part, round_params).with_columns(
            (pl.col("close").shift(-1) / pl.col("open").shift(-1) - 1)
            .alias("next_return")
        ).slice(warmup).drop_nulls(["momentum", "next_return"])
        for name, part in splits.items()
    }


def fit(train, round_params):
    estimator = make_pipeline(
        StandardScaler(), LogisticRegression(C=round_params["C"], random_state=0)
    )
    return estimator.fit(
        train.select("momentum"),
        (train["next_return"] > 0).to_numpy(),
    )


def backtest(validation, predictions):
    # Each signal enters at the next open and exits at that bar's close.
    # Fixed notional; each round trip pays 5 bps fee + 5 bps slip per side.
    net_returns = predictions * (validation["next_return"].to_numpy() - 0.002)
    return float(np.mean(net_returns))


def model(data, round_params):
    fitted = fit(data["train"], round_params)
    predictions = fitted.predict(data["validation"].select("momentum"))
    fitted_models[(round_params["window"], round_params["C"])] = fitted
    return {"validation_mean_return": backtest(data["validation"], predictions)}
```

### Run the experiment

Save the driver beside it as `run.py`. Supply your own `bars.csv` containing recorded, completed bars with unique UTC `datetime` values marking bar completion, and positive `open` and `close` prices. The loader and this schema belong to the example; your own SFD can prepare different data and features.

Allow enough history to retain observations after each split's 48-bar warm-up and forward-label removal, with both target classes in the prepared training data.

```python
import polars as pl
import research

from limen.experiment import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy

bars = (
    pl.read_csv("bars.csv", try_parse_dates=True)
    .with_columns(pl.col("datetime").dt.replace_time_zone("UTC"))
    .sort("datetime")
)
domain = ParamDomain(research.params())
uel = UniversalExperimentLoop(
    data=bars, sfd=research, search_strategy=GridStrategy(domain)
)
uel.run("my-research", n_permutations=6, post_processing=False)
```

Run the research with:

```bash
python run.py
```

You now have one results row per candidate in `uel.experiment_log`, also saved as `my-research.csv`. See [experiment results](https://docs.vaquum.fi/limen/guides/universal-experiment-loop#what-uel-stores-after-a-run) for the available outputs.

### Create a Sensor

Continue in `run.py`. Select the best validation result and wrap its retained fitted model in a native Sensor. Reuse the same feature function and fitted pipeline:

```python
from limen import Sensor

selected = (
    uel.experiment_log.sort("validation_mean_return", descending=True)
    .select(*research.params()).row(0, named=True)
)
sensor = Sensor.from_model(
    research.fitted_models[(selected["window"], selected["C"])],
    prepare=research.features,
    params=selected,
    selection_cutoff=bars["datetime"][int(len(bars) * 0.8)],
    output_id="next-bar-up",
)
prediction = sensor.predict(bars)
```

The Sensor reuses the fitted scaler and classifier without refitting, selects the feature names recorded during fitting, and blocks null or nonfinite feature values and decisions before the selection cutoff. Pass new completed history to `sensor.predict(history)` to get its latest `BarPrediction`: a 0/1 prediction or an explicit unavailable reason. Your Python still owns preparation and causality. Custom prediction and output mapping remain available, while `Sensor.from_python` accepts arbitrary fitted state and inference logic. See [Python Sensor construction](https://docs.vaquum.fi/limen/guides/trainer#python-construction).

### Create a sensor cohort

Your own Python selects any number of candidates from the results. Wrap their retained fitted models as above, then combine them with `Cohort.from_python(selected_sensors, aggregate="vote")` for this binary example. Call `cohort.predict(history)` just like a Sensor. Members declare the same `output_id` to identify compatible output meanings; Limen checks timestamp alignment and preserves unavailable results. Mean aggregation and custom aggregation functions are also available. See [Python cohorts](https://docs.vaquum.fi/limen/guides/cohort#python-cohorts) and the [worked Python workflow](https://docs.vaquum.fi/limen/guides/universal-experiment-loop#direct-python-run).

The example fits on the first 60% of observations and compares candidates on the next 20%, using identical evaluation timestamps. The final 20% are reserved for assessment after selection. Replace any research component with your own Python; the one-feature model and one-bar backtest are only the starting template.

Continue with [custom SFDs](https://docs.vaquum.fi/limen/guides/single-file-decoder#custom-sfds), [UEL](https://docs.vaquum.fi/limen/guides/universal-experiment-loop#direct-python-run), or [search and feedback](https://docs.vaquum.fi/limen/guides/advanced-search).

## Compiled Research

Declare the experiment and let Limen enforce its pipeline contract. The Manifest specifies the data, chronological splits, preparation, model and parameter space; the CLI validates and compiles that definition before executing it.

Start with a packaged logistic-regression experiment:

```bash
limen init logreg-first.yaml --template logreg_binary
```

<details>
<summary>Read the full YAML manifest</summary>

```yaml
schema_version: "1.0"

metadata:
  name: "logreg-first"
  limen_version: "5.42.1"
  mode: development
  description: Logistic regression binary classifier with calibration and feature perturbation

sfd:
  manifest:
    type: ml

    data_source:
      method: limen.data.HistoricalData.get_spot_klines
      params:
        kline_size: 3600

    split_dates:
      train_start: "2024-01-01"
      train_end:   "2024-09-30"
      val_start:   "2024-10-01"
      val_end:     "2024-11-30"
      test_start:  "2024-12-01"
      test_end:    "2024-12-31"

    indicators:
      - func: limen.indicators.roc
        params:
          period: "{roc_period}"
          group: momentum
      - func: limen.indicators.atr
        params:
          period: 14
          group: volatility
      - func: limen.indicators.ppo
        params:
          group: momentum
      - func: limen.indicators.wilder_rsi
        params:
          group: momentum

    features:
      - func: limen.features.fractional_diff
        params:
          d: "{frac_diff_d}"
          cols:
            - close
      - func: limen.features.vwap
        params:
          group: volume
      - func: limen.features.kline_imbalance
        params:
          group: volume

    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: "roc_{roc_period}"
        quantile: "{q}"
      transform_params:
        shift: "{shift}"

    scaler:
      from_params: scaler_type

    feature_ablation:
      drop_count_key: feature_drop_count
      seed_key: feature_drop_seed

    strict_mode: true

    reference_architecture: limen.sfd.reference_architecture.logreg_binary.logreg_binary

    calibration:
      probability_calibration:
        func: limen.calibration.sklearn_probability_calibrator
        params:
          method: "{cal_method}"
      threshold_function:
        func: limen.calibration.grid_threshold_optimizer
        params:
          metric: limen.metrics.balanced_metric.balanced_metric
          threshold_min: "{threshold_min}"
          threshold_max: "{threshold_max}"
          threshold_step: "{threshold_step}"

  params:
    # data prep
    frac_diff_d: [0.0]
    shift: [-1, -3]
    q: [0.45, 0.50]
    roc_period: [1, 12]
    # view perturbation
    scaler_type: [robust, rank_gauss]
    # feature perturbation
    feature_groups: [all, momentum]
    feature_drop_count: [0]
    feature_drop_seed: [42]
    # calibration
    use_calibration: [true]
    use_threshold: [true, false]
    cal_method: [sigmoid]
    threshold_min: [0.25]
    threshold_max: [0.65]
    threshold_step: [0.05]
    # classifier
    solver: [lbfgs, liblinear]
    penalty: [l2]
    dual: [false]
    tol: [0.01]
    C: [0.1, 1.0]
    fit_intercept: [true]
    intercept_scaling: [1.0]
    class_weight: [0.55]
    random_state: [42]
    max_iter: [120]
    multi_class: [deprecated]
    verbose: [0]
    warm_start: [false]
    n_jobs: [-1]
    l1_ratio: [null]

uel:
  n_permutations: 12
  search_strategy:
    type: random
  prep_each_round: true
  output_format: csv
```

</details>

Review the generated YAML, including its data dates, model settings and `uel.n_permutations` research budget. Then validate, profile, compile and run:

```bash
limen validate logreg-first.yaml
limen profile logreg-first.yaml
limen run --dry-run logreg-first.yaml
limen run logreg-first.yaml
```

Validation checks schema and references; dry run compiles without executing candidates. Execution applies the configured split, fitting, alignment and evaluation rules. A rejected configuration must be corrected before that workflow can proceed.

The CLI prints its result directory:

| Artifact | What it records |
|---|---|
| Copied YAML manifest | The experiment definition. |
| `metadata.json` | Run configuration and provenance metadata. |
| `results.csv` | Candidate results. |
| `round_data.jsonl.gz` | Round identities, parameters and payloads selected by the recording policy. |

Row evidence and recovery depend on the [recording policy](https://docs.vaquum.fi/limen/guides/universal-experiment-loop#recording-policy). Choose it before a run when you need later replay or continuation.

Continue with the [CLI workflow](https://docs.vaquum.fi/limen/guides/command-line-interface), [Manifest contract](https://docs.vaquum.fi/limen/guides/experiment-manifest), or [end-to-end example](https://docs.vaquum.fi/limen/guides/end-to-end-workflow).

## Shared safety

Both interfaces share the search engine and reusable research components. Python Research lets you compose or replace them; Compiled Research wires their safeguards into a checked pipeline.

### Common foundations

MSQ and UEL provide parameter-domain checks, parameter identities, explicit budgets, round results and convergence-warning records. Seeded built-in search supports repeatable candidate ordering. Optional persistence adds checkpoint and artifact-integrity checks; recovery requires a compatible checkpoint and full recording. Search seeds do not freeze models, data or environments. See [search and recovery](https://docs.vaquum.fi/limen/guides/advanced-search).

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

See the [preparation contract](https://docs.vaquum.fi/limen/guides/experiment-manifest#how-a-manifest-executes), [backtest mechanics](https://docs.vaquum.fi/limen/guides/backtest) and [inference contract](https://docs.vaquum.fi/limen/guides/trainer#sensor) for each control's scope.

### Additional validation, when enabled

- [Walk-forward validation](https://docs.vaquum.fi/limen/guides/experiment-manifest#walk-forward-sweeps) refits each fold with declared purge and embargo gaps. Choose gaps that cover your label horizon.
- A [validation-return objective](https://docs.vaquum.fi/limen/reference/calibration#shared-validation-return-objective) binds supported threshold tuning and reducers to validation execution. [DSR and PBO reports](https://docs.vaquum.fi/limen/guides/benchmark#walk-forward-selection-report) assess retained research evidence; failed or unavailable criteria are reported rather than stopping the run.
- Opt-in [project holdout reservations](https://docs.vaquum.fi/limen/guides/final-holdout) exclude reserved observations from supported research and replay. Active reservations reject custom SFDs and unsupported modes; see the workflow's restrictions.

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
| Python Research | [Search and feedback](https://docs.vaquum.fi/limen/guides/advanced-search), [custom SFDs](https://docs.vaquum.fi/limen/guides/single-file-decoder#custom-sfds), [UEL](https://docs.vaquum.fi/limen/guides/universal-experiment-loop). |
| Compiled Research | [CLI](https://docs.vaquum.fi/limen/guides/command-line-interface), [Manifest](https://docs.vaquum.fi/limen/guides/experiment-manifest), [end-to-end workflow](https://docs.vaquum.fi/limen/guides/end-to-end-workflow). |
| Research tools | [Features](https://docs.vaquum.fi/limen/reference/features), [targets](https://docs.vaquum.fi/limen/reference/targets), [backtesting](https://docs.vaquum.fi/limen/guides/backtest), [analysis](https://docs.vaquum.fi/limen/guides/log). |
| Model reuse | [Trainer](https://docs.vaquum.fi/limen/guides/trainer), [Cohort](https://docs.vaquum.fi/limen/guides/cohort). |

See the [documentation hub](https://docs.vaquum.fi/limen/overview/docs-hub), [Roadmap](https://docs.vaquum.fi/limen/overview/roadmap), and [Security Assurance Case](https://github.com/Vaquum/Limen/blob/main/docs/Developer/Security-Assurance-Case.md).

## Contributing

Contribution starts through [CONTRIBUTING.md](https://github.com/Vaquum/Limen/blob/main/CONTRIBUTING.md), [docs changes](https://github.com/Vaquum/Limen/tree/main/docs), or [open issues](https://github.com/Vaquum/Limen/issues).

Before contributing, start with the [Developer docs](https://docs.vaquum.fi/limen/developer).

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
