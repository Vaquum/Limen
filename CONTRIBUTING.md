# Contributing

Start with [AGENTS.md](AGENTS.md), the [repository laws](.github/REPOSITORY-LAW.md), and the [developer docs](docs/Developer/README.md). Changes to research examples and the reader follow the [documentation system contract](docs/Developer/Documentation-System.md).

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Canonical local validation:

```bash
python -m tests.run
python -m build
ruff check .
```

Docs-site checks live under `docs-site`:

```bash
cd docs-site
npm ci
npm run check
```

The candidate reader also requires current execution evidence from both projects. Follow the [reader validation procedure](docs/Developer/Documentation-System.md#validating-the-candidate-reader) for research dependencies, source provenance, wheel execution and browser checks. A retained renderer build alone does not validate the candidate reader.

## Pull Requests

Commit a useful increment, push it and open a non-draft PR before local checks. Then:

- read the relevant docs page for the touched subsystem
- update tests and docs with the behavior change
- leave version surfaces alone (version/changelog move only at release time)
- run `python3 -m tests.run`
- run `python3 -m build` when packaging metadata, package data, or README links change
- run `ruff check .` when Python source changes
- run docs-site checks when docs-site or public docs change

Use the issue templates for bug reports, feature requests, support requests, and security reports.

## Indicator and Feature Taxonomy

[datasets/taxonomy.json](datasets/taxonomy.json) is the shared discovery catalogue for indicators and features. Each public definition module has exactly one entry: `name` is its filename without `.py`, `type` is `indicator` or `feature`, and `category` is one primary market behavior:

| Category | Meaning |
|---|---|
| `trend` | Directional bias and persistence. |
| `momentum` | Price velocity, impulse and directional oscillators. |
| `volatility` | Dispersion, range magnitude and risk statistics. |
| `volume` | Trading activity, flow and liquidity. |
| `market_structure` | Price levels, candle geometry and relative position. |
| `trade_logic` | Entry decisions, stops, targets and execution eligibility. |
| `temporal` | Calendar context, lagging and general statistical transforms. |
| `exotic` | Experimental mathematics outside the other categories. |

Classify the output's meaning. A directional breakout belongs to `momentum`; a range magnitude belongs to `volatility`. Regime outputs use the behavior they describe. Candle patterns and `close_position` describe geometry, so belong to `market_structure`; `volatility_barriers` produces TP/SL distances, so belongs to `trade_logic`. `conserved_flux_renormalization` uses `exotic` for its experimental multiscale flux/entropy fingerprint.

Adding, renaming or removing a public module under `limen/indicators` or `limen/features` requires the corresponding taxonomy update in the same PR. Private modules beginning with `_` are excluded. Multiple functions in one module share its entry; the `(type, name)` pair identifies a definition. Frontends and other consumers read the JSON directly; installed distributions also provide it under `share/limen/datasets`.

Run `python -m pytest tests/test_feature_taxonomy.py`. The existing required `PR Checks Tests` gate runs this check and rejects missing, duplicate, unknown or invalid-category entries.
