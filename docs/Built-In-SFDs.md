# Built-in SFDs

Limen ships model templates under `limen/yaml/templates` and their component implementations under `limen.sfd.foundational_sfd`. Compiled Research edits YAML and executes it through the CLI. The [Compiled project](examples/compiled-research/README.md) supplies recorded-data declarations for those model families. The [Python project](examples/python-research/README.md) owns its fitting, scoring and retained state directly.

They show the packaged Limen experiment shape because each one combines:

- `params()`
- `manifest()`
- a matching reference-architecture model surface

## Prerequisites

- `vaquum-limen[data]` for bundled data sources
- the matching optional extra for LightGBM/XGBoost, SciPy-backed DLinear, or TabPFN runs
- the CLI first-run sequence from [Command-Line Interface](Command-Line-Interface.md)

## The current catalog

| SFD | Task shape | Notes |
|---|---|---|
| `logreg_binary` | binary classification | canonical manifest-driven logistic-regression reference flow |
| `lightgbm_binary` | binary classification | the tradeline long-binary experiment: line-geometry features, train-fitted breakout target, LightGBM classifier |
| `gaussian_hmm_binary` | binary next-return sign | causal window-local homogeneous Gaussian return model; requires `hmm` |
| `random_binary` | binary classification baseline | sanity-check and control-comparison flow |
| `ridge_regressor` | regression | sklearn Ridge with lagged returns and train-fitted robust scaling |
| `xgboost_regressor` | regression | tree-based regression workflow |
| `dlinear_regressor` | regression | canonical DLinear decomposition-linear reference, deterministic closed-form fit |
| `tabpfn_binary` | binary classification | lazy symbols are always importable; model use requires `tabpfn` |
| `rule_based` | rule-based long/flat | predicate-driven strategy with no learned model |
| `common_strategies` | rule-based long/flat baselines | eight fixed passive, trend and mean-reversion rules; see [Common Strategies](Common-Strategies.md) |
| `dollar_bar_crash_reversal` | rule-based long/flat | exhaustive crash-and-maker-flow reversal sweep on `$15M` dollar bars |

## Foundational SFD versus reference architecture

Built-in ML SFDs have matching model modules in [Reference Architecture](Reference-Architecture.md). Rule-based SFDs share `RuleBasedStrategy`; `dollar_bar_crash_reversal` does not introduce another model class.

The split is:

| Layer | Owns |
|---|---|
| foundational SFD | search space plus manifest pipeline |
| reference architecture | class-based model contract and function wrapper |

For the logistic-regression SFD:

- `limen.sfd.foundational_sfd.logreg_binary` owns the packaged experiment
- `limen.sfd.reference_architecture.logreg_binary` owns the model implementation

This separation is what lets [Trainer](Trainer.md) reconstruct and replay a finished experiment with the matching `ReferenceModel`.

## `dollar_bar_crash_reversal`

`dollar_bar_crash_reversal` is a fixed-protocol intraday research sweep over native BTCUSDT `$15M` dollar bars. It combines four-hour log momentum with a robust maker-flow deviation and converts each trigger into a wall-clock long position.

The bundled YAML template fixes:

- data window: `2020-02-01` through `2026-07-10`
- splits: train before `2024-01-01`, validation during 2024, test from `2025-01-01`
- grid: four momentum thresholds × four flow thresholds × five holds (`30`, `45`, `60`, `90`, `120` minutes), all 80 combinations
- execution: one-bar lag, 10 bps fee plus 5 bps slippage on each entry and exit fill

The canonical projects do not claim to replay this entire multi-year sweep. The recorded [crash-reversal fixture regression](../tests/test_dollar_bar_crash_reversal.py) separately checks transform availability, signal counts and exact native evaluation on its pinned observations. Treat the packaged sweep's fixed dates, costs and trigger availability as part of its research protocol, rather than an expected live return.

The transform prevents the final observed row of each UTC day from initiating a trigger because same-date membership requires the next row. A position already within its wall-clock hold can remain active on that row. Treat trigger initiation as one-row available; the default execution lag adapts the research signal to the backtest but does not make the raw trigger same-row causal.

## `logreg_binary`

`logreg_binary` is the standard manifest-driven binary classifier in the package.

It combines:

- indicators such as `roc`, `atr`, `ppo`, and `wilder_rsi`
- features such as `vwap` and `kline_imbalance`
- a fitted quantile-based target
- scaler selection from params (`logreg`, `robust`, `rank_gauss`)
- `strict_mode=True` — unexpected mid-split nulls abort the round and record an error in `results.csv`
- the `LogRegBinary` reference model
- `CalibrationBuilder` with `sklearn_probability_calibrator` and `grid_threshold_optimizer`

The classifier parameter surface mirrors the sklearn `LogisticRegression` constructor through manifest params: `solver`, `penalty`, `dual`, `tol`, `C`, `fit_intercept`, `intercept_scaling`, `class_weight`, `random_state`, `max_iter`, `multi_class`, `verbose`, `warm_start`, `n_jobs`, and `l1_ratio`.

The calibration search space includes `use_calibration`, `use_threshold`, `cal_method`, `threshold_min`, `threshold_max`, and `threshold_step`, creating a grid of calibration modes within a single experiment run.

The Compiled project's `selection` and `calibration` declarations use a smaller ROC-based logistic pipeline. Their counts describe those declarations, rather than the complete packaged feature set.

## `lightgbm_binary`

`lightgbm_binary` is the tradeline long-binary experiment: the line-geometry research track packaged on Limen rails.

It combines:

- the grouped line transforms `price_lines` and `quantile_price_lines` with swept geometry (`max_duration_hours`, `min_height_pct`, `quantile_threshold`)
- context from `roc`, `distance_from_high`/`distance_from_low`/`price_range_position`, `parkinson_volatility`/`volatility_ratio`, and `cyclical_time_features`
- the train-fitted `TradelineLongBinaryTarget` (confirmed-breakout label from a line-height percentile threshold)
- scaler selection from params (`robust`, `rank_gauss`, `logreg`) and feature ablation
- `strict_mode=True`
- the `LightGBMBinary` reference model with the full `LGBMClassifier` parameter surface and early stopping
- `CalibrationBuilder` with `sklearn_probability_calibrator` and `grid_threshold_optimizer`
- swept backtest economics (`fee_bps`, `slip_bps`)

The matching YAML template is `limen/yaml/templates/lightgbm_binary.yaml` (`limen init my_experiment.yaml --template lightgbm_binary`).

This SFD keeps the line-context family live-safe by setting `include_research_only: false`, which omits `active_lines` and `active_quantile_count`. Those span-count outputs are not live-computable and require explicit research-only opt-in.

## `ridge_regressor`

`ridge_regressor` is a linear, L2-regularized next-return baseline. The Python
SFD and `limen/yaml/templates/ridge_regressor.yaml` combine one-bar returns,
lags 1 through 5, a train-fitted `RobustScaler`, `NextReturnTarget(periods=1,
scale=100.0)`, and `strict_mode=True`. The small starter space varies `alpha`
and `fit_intercept`; all eight wrapper parameters are declared explicitly.

Start from `limen init ridge.yaml --template ridge_regressor`. The model uses
scikit-learn, which is already a core dependency. No alternate execution path
is required: YAML experiments, logged metrics, Trainer reconstruction, and
Sensor inference use the ordinary Limen interfaces. See
[Reference Architecture](Reference-Architecture.md#ridgeregressor) for the
model and solver contract.

## `random_binary`

`random_binary` is the baseline binary classifier. It is stochastic and intended for control runs, smoke tests, and low-skill comparison points.

The Compiled project executes this family as a stochastic baseline and checks the recorded prediction domain and metric range. It does not claim deterministic Trainer replay for that baseline.

Because it is stochastic, it is a poor fit for deterministic reconstruction in [Trainer](Trainer.md).

## `xgboost_regressor`

`xgboost_regressor` predicts percentage forward returns using `NextReturnTarget`.
It requires the `boosting` extra (`pip install 'vaquum-limen[boosting]'`).

The Python SFD searches the target horizon, return/volatility/volume windows,
stochastic/RSI settings, price SMAs, range/Bollinger settings, volume regime,
lag, feature groups, scaler and feature ablation. `params()` lists the exact
candidate values. Model controls use the wrapper's defaults unless the search
space overrides them; the default sweep uses CPU histogram trees and one native
thread. `device` and `tree_method` reach the model through the stock manifest.

Feature groups are `momentum`, `trend`, `volume`, `volatility`, and `structure`.
Select `all` or pipe-delimited names such as `momentum|trend`. Each group includes
its own lagged research columns. The one-bar return and its configured lag form
a shared baseline. Scaling is fitted on training data; ablation uses the existing
`feature_drop_count` and `feature_drop_seed` controls. Prepare each sampled round
because the feature pipeline and target can change.

The Compiled project includes XGBoost in its recorded model-family comparison:

<!-- example: builtins-compiled-model-families source=docs/examples/compiled-research/run.sh region=models scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar \
  compiled-research/common_strategies.yaml \
  compiled-research/family_ridge_regressor.yaml \
  compiled-research/family_dlinear_regressor.yaml \
  compiled-research/family_xgboost_regressor.yaml \
  compiled-research/family_random_binary.yaml \
  compiled-research/family_direction_sizing.yaml
```

This diagnostic uses one declared recorded-data case per family (two direction/sizing cases), retains native prediction payloads. The independent assertions recompute continuous-family RMSE from the original aligned labels. It establishes these declared cases, rather than the full packaged search spaces.

The separate [XGBoost sweep driver](examples/xgboost_sweep.py) remains available for broader research. Its `test` mode selects a 5,000-row two-hour source; `prod` selects hourly observations from 2025-01-01. Its seed controls candidate sampling. CUDA additionally needs a compatible XGBoost build and GPU; inspect the fitted booster configuration to establish the actual backend. Those broader and GPU runs are outside the canonical project's recorded proof.

## `dlinear_regressor`

`dlinear_regressor` is the canonical DLinear reference experiment: Limen's gold standard for DLinear semantics, intended as the parity anchor for downstream comparisons.

It combines:

- `window_return(period=1)` to form the 1-bar close-to-close return series `ret_1`
- `lag_range(col='ret_1', start=0, end=lookback_end)` to build the lookback window as feature columns
- `NextReturnTarget` with a sweepable `horizon` (percentage return over the next `horizon` bars)
- `strict_mode=True`
- the `DLinearRegressor` reference model

`params()` exposes `lookback_end` (window length minus one), `kernel_size` (odd moving-average kernel of the decomposition), `alpha` (ridge strength on the component heads), and `horizon`.

The fit is closed-form, so every round is deterministic with no seed and the full sweep runs at interactive speed. The matching YAML template is `limen/yaml/templates/dlinear_regressor.yaml` (`limen init my_experiment.yaml --template dlinear_regressor`).

It requires `scipy` (the `stats` extra), loaded lazily inside the model.

## `tabpfn_binary`

`tabpfn_binary` is an optional packaged SFD. Its lazy module and exported symbols can be imported without TabPFN, but model construction and training require the `tabpfn` extra. That dependency is intentionally outside the base install because it is materially larger than the default sklearn path.

It uses `CalibrationBuilder` with the same probability calibration and threshold optimization wiring as `logreg_binary`, so its results also include `optimal_threshold` and `val_score` when calibration is active.

That optional status matters at execution time: discovery and template listing work in the base environment, while a TabPFN run fails until the extra is installed.

The opt-in foundational-SFD test runs one CPU ensemble on 384 recorded hourly
Bitcoin bars, checks the actual fitted classifier and verifies its predictions,
probabilities, accuracy and AUC. A second case verifies nondefault scalar, list
and configuration candidates through a compiled run and its saved parameters.
From a checkout with the `test,data,tabpfn`
extras installed, run
`LIMEN_RUN_TABPFN_MODEL_TEST=1 python -m pytest tests/test_tabpfn.py -q`.

The provisioned baseline is Python 3.12 on macOS ARM64 with TabPFN 6.4.1;
the [real-model campaign](examples/compiled-research/README.md#optional-model-campaign)
records the dependency environment and immutable v2 checkpoint download with
its SHA-256. Put that checkpoint in the TabPFN model cache, selected with
`TABPFN_MODEL_CACHE_DIR`, or run from its directory. A missing checkpoint requires
upstream network access; market observations come from the repository fixture.
An explicit opt-in fails if the dependency, checkpoint or execution is unavailable.
Ordinary CI skips this model test and validates the separate retained campaign.

## Running one immediately

The Compiled selection entry point validates the YAML, runs its declared search and assesses the selected recorded rounds:

<!-- example: builtins-compiled-selection source=docs/examples/compiled-research/run.sh region=selection scenario=selection interface=compiled role=executed -->
```bash
limen validate compiled-research/selection.yaml
limen profile compiled-research/selection.yaml
limen run --dry-run compiled-research/selection.yaml
limen run --no-progress-bar compiled-research/selection.yaml
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
```

The [CLI lifecycle](Command-Line-Interface.md#first-yaml-run) verifies template scaffolding and installed commands. Declare the data source and `uel.search_strategy` in YAML. The CLI compiles that declaration and fetches its input through the configured source.

## How to choose

- Choose `logreg_binary` for the canonical Limen path.
- Choose `random_binary` for a baseline or smoke-test decoder.
- Choose `xgboost_regressor` for continuous targets that should use tree-based regression.
- Choose `dlinear_regressor` for a deterministic linear forecasting reference with canonical DLinear semantics.
- Choose `tabpfn_binary` only when that dependency is installed and the TabPFN workflow is required.
- Choose `dollar_bar_crash_reversal` for the bundled exhaustive dollar-bar rule sweep and its fixed-cost evidence contract.

## Read next

- Continue to [Single-File Decoder](Single-File-Decoder.md) for the general SFD contract.
- Continue to [Command-Line Interface](Command-Line-Interface.md) for the YAML run loop.
- Continue to [Reference Architecture](Reference-Architecture.md) for the class-based model layer underneath these built-in decoders.
- Continue to [Experiment Manifest](Experiment-Manifest.md) to adapt one of these into a custom SFD.


## `gaussian_hmm_binary`

The native manifest supplies hourly returns with 12/24-bar lookbacks and a
bounded 2048-bar source. The YAML template bounds its input to 1024 hourly bars
and eight random rounds over 16 combinations. Both use the observed next-close
sign target. Native controls and probability semantics are documented in
[Reference Architecture](Reference-Architecture.md#gaussianhmmbinary).
