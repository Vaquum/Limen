<div align="center">
  <br />
  <a href="https://github.com/Vaquum"><img src="https://github.com/Vaquum/Home/raw/main/assets/Logo.png" alt="Vaquum" width="150" /></a>
  <br />
</div>
<br>
<div align="center"><strong>Vaquum Limen turns Bitcoin market data into searchable alpha, backtested signals, and decoder cohorts.</strong></div>

<div align="center">
  <a href="#limen">Limen</a> •
  <a href="#what-limen-is-not">What Limen Is Not</a> •
  <a href="#capabilities">Capabilities</a> •
  <a href="#first-experiment">First Experiment</a> •
  <a href="#learn-more">Learn More</a>
</div>
<br>
<div align="center">
  <a href="https://www.bestpractices.dev/projects/11898"><img src="https://www.bestpractices.dev/projects/11898/badge" alt="OpenSSF Best Practices" /></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/Vaquum/Limen"><img src="https://api.scorecard.dev/projects/github.com/Vaquum/Limen/badge" alt="OpenSSF Scorecard" /></a>
  <a href="https://github.com/Vaquum/Limen/tree/python-coverage-comment-action-data"><img src="https://raw.githubusercontent.com/Vaquum/Limen/python-coverage-comment-action-data/badge.svg" alt="Coverage" /></a>
</div>

<hr />

# Limen

Limen is a manifest-driven Bitcoin alpha research engine. It turns market data into searchable experiments, benchmark-style analytics, backtest results, and decoder cohorts.

Limen brings data preparation, feature construction, target shaping, parameter search, and post-run analysis into one Python workflow. It supports both machine learning and rule-based research, but it stays focused on signal research rather than decisioning or execution.

## What Limen Is Not

Limen is not:

- a trade execution system
- a downstream trade decision engine
- a generic multi-asset research platform

In the wider Vaquum architecture, Origo sits upstream as the data layer. Nexus, Praxis, and Veritas sit downstream for decisioning, execution, and oversight.

## Capabilities

- Manifest-driven experiment pipelines
- Search across models, rules, features, targets, and hyperparameters
- Extensive built-in indicator and feature library for Bitcoin research
- Support for both machine learning and rule-based strategy research
- Bitcoin-native transforms, scaling, and target construction
- Leakage-safe train, validation, and test workflows
- Built-in backtesting, confusion analytics, and parameter diagnostics
- Decoder cohort construction and regime-diversified model pooling
- Reproducible runs with checkpointing, resumption, and retraining

## First Experiment

The fastest first success is a small parameter sweep on the bundled BTC/USDT kline dataset with the built-in logistic-regression decoder.

1. Install the package:

```bash
pip install vaquum_limen
```

2. Load data and run a first experiment:

```python
import polars as pl
import limen

data = pl.read_csv(
    "https://raw.githubusercontent.com/Vaquum/Limen/refs/heads/main/datasets/klines_2h_2020_2025.csv",
    try_parse_dates=True,
)

uel = limen.UniversalExperimentLoop(data=data, sfd=limen.sfd.logreg_binary)

uel.run(
    experiment_name="logreg-first",
    n_permutations=25,
    prep_each_round=True,
)
```

3. Inspect the core outputs:

- `uel.experiment_log` for the parameter sweep results
- `uel.experiment_confusion_metrics` for confusion analytics
- `uel.experiment_backtest_results` for backtest results

That path is the simplest way to get a real Limen run on your machine. If you want richer run directories, checkpoints, resumability, and stored round artefacts, continue into the UEL documentation below.

## Learn More

- Start with the full docs hub in [docs/README.md](docs/README.md)
- Define research units in [docs/Single-File-Decoder.md](docs/Single-File-Decoder.md), [docs/Built-In-SFDs.md](docs/Built-In-SFDs.md), and [docs/Experiment-Manifest.md](docs/Experiment-Manifest.md)
- Run experiments in [docs/Universal-Experiment-Loop.md](docs/Universal-Experiment-Loop.md) and extend the artifact-rich path through [docs/Advanced-Search.md](docs/Advanced-Search.md) and [docs/Reducers-And-Feedback.md](docs/Reducers-And-Feedback.md)
- Analyze results in [docs/Log.md](docs/Log.md), [docs/Benchmark.md](docs/Benchmark.md), and [docs/Backtest.md](docs/Backtest.md)
- Understand the model layer in [docs/Reference-Architecture.md](docs/Reference-Architecture.md) and the helper layer in [docs/Utilities.md](docs/Utilities.md)
- Promote finished runs into reusable outputs with [docs/Trainer.md](docs/Trainer.md) and [docs/Regime-Diversified-Opinion-Pools.md](docs/Regime-Diversified-Opinion-Pools.md)
- Contribute through [docs/Developer/README.md](docs/Developer/README.md)

## Contributing

The simplest way to start contributing is by [joining an open discussion](https://github.com/Vaquum/Limen/issues?q=is%3Aissue%20state%3Aopen%20label%3Aquestion%2Fdiscussion), contributing to [the docs](https://github.com/Vaquum/Limen/tree/main/docs), or by [picking up an open issue](https://github.com/Vaquum/Limen/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug%20OR%20label%3Aenhancement%20OR%20label%3A%22good%20first%20issue%22%20OR%20label%3A%22help%20wanted%22%20OR%20label%3APriority%20OR%20label%3Aprocess).

Before contributing, start with [docs/Developer/README.md](docs/Developer/README.md).

## Vulnerabilities

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/Vaquum/Limen/security/advisories/new).

## Citations

If you use Limen for published work, please cite:

Vaquum Limen [Computer software]. (2026). Retrieved from https://github.com/Vaquum/Limen.

## License

[MIT License](https://github.com/Vaquum/Limen/blob/main/LICENSE).
