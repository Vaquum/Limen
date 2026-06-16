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
- Split-first train, validation, and test workflows
- Built-in benchmark, backtest, and parameter diagnostics
- Decoder cohort construction with pluggable selection
- Reproducible runs with checkpointing, resumption, and retraining

## First Experiment

The first runnable path is a YAML manifest executed through the `limen` CLI.

1. Install the package:

```bash
pip install vaquum-limen
```

2. Scaffold a starter manifest:

```bash
limen init logreg-first.yaml --template logreg_binary
```

3. Validate, profile, and dry-run the manifest:

```bash
limen validate logreg-first.yaml
limen profile logreg-first.yaml
limen run --dry-run logreg-first.yaml
```

4. Run it:

```bash
limen run logreg-first.yaml
```

5. Inspect the result directory printed by the CLI:

- copied YAML manifest
- `metadata.json`
- `results.csv`
- `round_data.jsonl`

That path runs the manifest-backed engine without Python orchestration code. The Python API remains available for custom SFDs, custom prep/model logic, and direct UEL integration.

## Risk Boundary

Limen is research software. Benchmark and backtest outputs are not investment advice, trading advice, execution simulation, or a promise of future performance.

## Learn more

- Start with the full docs hub at [docs.vaquum.fi/limen](https://docs.vaquum.fi/limen/)
- Start with the YAML/CLI path in [Command Line Interface](https://docs.vaquum.fi/limen/reference/command-line-interface) and [Experiment Manifest](https://docs.vaquum.fi/limen/guides/experiment-manifest)
- Use [Universal Experiment Loop](https://docs.vaquum.fi/limen/guides/universal-experiment-loop) for the engine beneath CLI and direct Python integration
- Define extension research units in [Single-File Decoder](https://docs.vaquum.fi/limen/guides/single-file-decoder) and [Built-In SFDs](https://docs.vaquum.fi/limen/guides/built-in-sfds)
- Analyze results in [Log](https://docs.vaquum.fi/limen/guides/log), [Benchmark](https://docs.vaquum.fi/limen/guides/benchmark), and [Backtest](https://docs.vaquum.fi/limen/guides/backtest)
- Promote finished runs into reusable outputs with [Trainer](https://docs.vaquum.fi/limen/guides/trainer) and [Cohort](https://docs.vaquum.fi/limen/guides/cohort)
- Contribute through [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/Developer/README.md](docs/Developer/README.md)

## Contributing

Contribution starts through [CONTRIBUTING.md](CONTRIBUTING.md), [docs changes](https://github.com/Vaquum/Limen/tree/main/docs), or [open issues](https://github.com/Vaquum/Limen/issues).

Before contributing, start with [docs/Developer/README.md](docs/Developer/README.md).

## Support

Use [SUPPORT.md](SUPPORT.md) for support routes and scope boundaries.

## Vulnerabilities

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/Vaquum/Limen/security/advisories/new). Do not report vulnerabilities through public issues.

## Citations

Published work should cite:

Vaquum Limen [Computer software]. (2026). Retrieved from https://github.com/Vaquum/Limen.

Machine-readable citation metadata lives in [CITATION.cff](CITATION.cff).

## License

[MIT License](https://github.com/Vaquum/Limen/blob/main/LICENSE).
