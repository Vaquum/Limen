<h1 align="center">
  <br>
  <a href="https://github.com/Vaquum"><img src="https://github.com/Vaquum/Home/raw/main/assets/Logo.png" alt="Vaquum" width="150"></a>
  <br>
</h1>

<h3 align="center">Vaquum Limen is for parametric Bitcoin research and trading signal generation.</h3>

<p align="center">
  <a href="#value-proposition">Value Proposition</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>
<hr>

# Value Proposition

Vaquum Limen reduces complex and otherwise out-of-reach research, model development, and trading signal workflows into one cohesive Python API, powering Bitcoin quants with unparalleled ergonomics and productivity. Limen does not execute trades, and can be used as a source of alpha with any trading system.

# Quick Start

Anyone, with basic Python skills, can go from from nothing to robust tradeable signals in minutes.

1) Start by installing the package:

`pip install vaquum_limen`

2) Once the package is installed, you're ready to start your first alpha experiment. First, let's get some OHLC data to work with:

```
import polars

data_path = 'https://raw.githubusercontent.com/Vaquum/Limen/refs/heads/main/datasets/klines_2h_2020_2025.csv'

data = polars.read_csv(data_path, try_parse_dates=True)
```

3) Then, let's use that BTC/USDT data to perform a parameter sweep of 100 permutations using a Logistic Regression binary decoder:

```
import limen

uel = limen.UniversalExperimentLoop(data=data, sfd=limen.sfd.logreg_binary)

uel.run(experiment_name=f"LogReg-First",
        n_permutations=100, 
        prep_each_round=True)
```

Completing the parameter sweep yields several useful datasets in the `uel.object`: 

- Parameter Sweep Log
- Advanced Confusion Matrix
- Bactest Results

# Contributing

The simplest way to start contributing is by [joining an open discussion](https://github.com/Vaquum/Limen/issues?q=is%3Aissue%20state%3Aopen%20label%3Aquestion%2Fdiscussion), contributing to [the docs](https://github.com/Vaquum/Limen/tree/main/docs), or by [picking up an open issue](https://github.com/Vaquum/Limen/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug%20OR%20label%3Aenhancement%20OR%20label%3A%22good%20first%20issue%22%20OR%20label%3A%22help%20wanted%22%20OR%20label%3APriority%20OR%20label%3Aprocess).

**Before contributing, make sure to start by reading through the** [Contributing Guidelines](https://github.com/Vaquum/Limen/tree/main/docs/Developer).

# Citations

If you use Limen for published work, please cite:

Vaquum Limen [Computer software]. (2026). Retrieved from http://github.com/vaquum/limen.

# License

[MIT License](https://github.com/Vaquum/Limen/blob/main/LICENSE).
