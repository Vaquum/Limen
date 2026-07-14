<div align="center">
  <br />
  <a href="https://github.com/Vaquum"><img src="https://github.com/Vaquum/Home/raw/main/assets/Logo.png" alt="Vaquum" width="150" /></a>
  <br />
</div>
<br />
<div align="center"><b>Vaquum Limen turns Bitcoin market data into searchable signals, backtested outcomes, and decoder cohorts.</b></div>

<div align="center">
  <a href="#limen">Limen</a> •
  <a href="#what-limen-is-not">What Limen Is Not</a> •
  <a href="#capabilities">Capabilities</a> •
  <a href="#first-experiment">First Experiment</a> •
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

# Limen — Research engine

*Manifest-driven Bitcoin alpha research engine that turns market data into searchable signals, backtested outcomes, and decoder cohorts.*

Limen unifies parameter search across machine learning and rule-based strategies. Built-in analytics record benchmark, backtest, and cohort artifacts for inspection. The project evolves from Talos, a hyperparameter optimization framework for TensorFlow and Keras.

## What Limen Is Not

Limen is not:

- a trade execution system
- a downstream trade decision engine
- a generic multi-asset research platform

In the wider Vaquum architecture, Origo sits upstream as the data layer. Nexus, Praxis, and Veritas sit downstream for decisioning, execution, and oversight.

## Capabilities

- Manifest-driven experiment pipelines
- Search across models, rules, features, targets, and hyperparameters
- Built-in indicator and feature library for Bitcoin research
- Support for both machine learning and rule-based strategy research
- Bitcoin-native transforms, scaling, and target construction
- Perturbation-based robustness search across feature groups, scalers, and ablation
- Split-first train, validation, and test workflows
- Built-in benchmark, backtest, and parameter diagnostics
- Decoder cohort construction with pluggable selection
- Reproducible runs with checkpointing, resumption, and validated round replay

## First Experiment

The first runnable path is a YAML manifest executed through the `limen` CLI.

1. Install the package:

```bash
pip install "vaquum-limen[data]"
```

Supported runtime: Limen requires Python `>=3.10,<3.14`; package metadata advertises Python 3.10-3.13 on macOS and Linux. The first experiment reads Arrow data, so it needs the `data` extra. The default install is intentionally light for API surfaces that do not load data. Use `vaquum-limen[boosting]` for LightGBM/XGBoost models, `vaquum-limen[indicators]` for TA-Lib comparison tooling, `vaquum-limen[stats]` for statistical helpers, or `vaquum-limen[all]` for the full research stack. Security support covers the latest released Limen version through [SECURITY.md](https://github.com/Vaquum/Limen/blob/main/SECURITY.md).

1. Scaffold a starter manifest:

```bash
limen init logreg-first.yaml --template logreg_binary
```

1. Validate, profile, and dry-run the manifest:

```bash
limen validate logreg-first.yaml
limen profile logreg-first.yaml
limen run --dry-run logreg-first.yaml
```

1. Run it:

```bash
limen run logreg-first.yaml
```

1. Inspect the result directory printed by the CLI:

- copied YAML manifest
- `metadata.json`
- `results.csv`
- `round_data.jsonl`

That path runs the manifest-backed engine without Python orchestration code. The Python API remains available for custom SFDs, custom prep/model logic, and direct UEL integration.

## Risk Boundary

Limen is research software. Benchmark and backtest outputs are not investment advice, trading advice, execution simulation, regulatory approval, or a promise of future performance. Past performance is not predictive, and trading digital assets can result in total loss of capital.

## Learn more

- Start with the full [documentation hub](https://docs.vaquum.fi/limen/overview/docs-hub)
- Start with the YAML/CLI path in [Command-Line Interface](https://docs.vaquum.fi/limen/guides/command-line-interface) and [Experiment Manifest](https://docs.vaquum.fi/limen/guides/experiment-manifest)
- Use [Universal Experiment Loop](https://docs.vaquum.fi/limen/guides/universal-experiment-loop) for the engine beneath CLI and direct Python integration
- Define extension research units in [Single-File Decoder](https://docs.vaquum.fi/limen/guides/single-file-decoder) and [Built-In SFDs](https://docs.vaquum.fi/limen/guides/built-in-sfds)
- Strengthen the research surface with [Perturbation Strategies](https://docs.vaquum.fi/limen/guides/perturbation-strategies) for robustness, [Fractional Differentiation](https://docs.vaquum.fi/limen/guides/fractional-differentiation) for stationary features, and [Triple-Barrier Method](https://docs.vaquum.fi/limen/guides/triple-barrier-method) for path-dependent labels
- Analyze results in [Log](https://docs.vaquum.fi/limen/guides/log), [Benchmark](https://docs.vaquum.fi/limen/guides/benchmark), and [Backtest](https://docs.vaquum.fi/limen/guides/backtest)
- Promote finished runs into reusable outputs with [Trainer](https://docs.vaquum.fi/limen/guides/trainer) and [Cohort](https://docs.vaquum.fi/limen/guides/cohort)
- Contribute through [CONTRIBUTING.md](https://github.com/Vaquum/Limen/blob/main/CONTRIBUTING.md) and [Developer docs](https://docs.vaquum.fi/limen/developer)

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
