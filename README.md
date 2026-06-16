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
</div>

<hr />

<a id="limen"></a>

# Limen — Research engine

*Manifest-driven Bitcoin alpha research engine that turns market data into searchable signals, backtested outcomes, and decoder cohorts.*

Limen unifies parameter search across machine learning and rule-based strategies. Built-in analytics connect experiment outputs to benchmark, backtest, and cohort workflows. The project evolves from Talos, a hyperparameter optimization framework for TensorFlow and Keras.

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
- Leakage-safe train, validation, and test workflows
- Built-in backtesting, confusion analytics, and parameter diagnostics
- Decoder cohort construction with pluggable selection
- Reproducible runs with checkpointing, resumption, and retraining

## First Experiment

The first runnable path is a YAML manifest executed through the `limen` CLI.

1. Install the package:

```bash
pip install vaquum_limen
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

## Learn more

- Start with the full docs hub in [docs/README.md](docs/README.md)
- Start with the YAML/CLI path in [docs/Command-Line-Interface.md](docs/Command-Line-Interface.md) and [docs/Experiment-Manifest.md](docs/Experiment-Manifest.md)
- Use [docs/Universal-Experiment-Loop.md](docs/Universal-Experiment-Loop.md) for the engine beneath CLI and direct Python integration
- Define extension research units in [docs/Single-File-Decoder.md](docs/Single-File-Decoder.md) and [docs/Built-In-SFDs.md](docs/Built-In-SFDs.md)
- Analyze results in [docs/Log.md](docs/Log.md), [docs/Benchmark.md](docs/Benchmark.md), and [docs/Backtest.md](docs/Backtest.md)
- Understand the model layer in [docs/Reference-Architecture.md](docs/Reference-Architecture.md) and the helper layer in [docs/Utilities.md](docs/Utilities.md)
- Promote finished runs into reusable outputs with [docs/Trainer.md](docs/Trainer.md) and [docs/Cohort.md](docs/Cohort.md)
- Contribute through [docs/Developer/README.md](docs/Developer/README.md)

## Contributing

Contribution starts through [open discussions](https://github.com/Vaquum/Limen/issues), [docs changes](https://github.com/Vaquum/Limen/tree/main/docs), or [open issues](https://github.com/Vaquum/Limen/issues).

Before contributing, start with [docs/Developer/README.md](docs/Developer/README.md).

## Vulnerabilities

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/Vaquum/Limen/security/advisories/new).

## Citations

Published work should cite:

Vaquum Limen [Computer software]. (2026). Retrieved from https://github.com/Vaquum/Limen.

## License

[MIT License](https://github.com/Vaquum/Limen/blob/main/LICENSE).
