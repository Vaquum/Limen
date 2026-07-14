# Built-in SFDs

Limen ships foundational SFDs under `limen.sfd.foundational_sfd`; most have matching YAML templates under `limen/yaml/templates`. Ordinary runs should start from a YAML template when one exists; the Python modules are the packaged decoder layer beneath that path.

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
| `random_binary` | binary classification baseline | sanity-check and control-comparison flow |
| `xgboost_regressor` | regression | tree-based regression workflow |
| `dlinear_regressor` | regression | canonical DLinear decomposition-linear reference, deterministic closed-form fit |
| `tabpfn_binary` | binary classification | lazy symbols are always importable; model use requires `tabpfn` |
| `rule_based` | rule-based long/flat | predicate-driven strategy with no learned model |
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

The reference candidate (`-575` bps momentum, `-0.5` flow score, `60` minutes) produced sparse executed samples in the fixed window: 129 train, 13 validation, and 14 test trades. Mean compounded net PnL per executed trade was 75.7, 116.1, and 127.1 bps respectively after costs. On the irregular test bars, observed position paths spanned about 61.6–149.7 minutes. These are reproducible backtest observations, not a population guarantee or expected live return.

The transform excludes the final observed row of each UTC day because same-date membership requires the next row. Treat this as one-row availability; the default execution lag adapts the research signal to the backtest but does not make the raw row same-row causal.

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

In the bundled smoke path, it prepared:

- `24` training features
- `3610` training rows

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

## `random_binary`

`random_binary` is the baseline binary classifier. It is stochastic and intended for control runs, smoke tests, and low-skill comparison points.

In the bundled smoke path, it prepared:

- `18` training features
- `2999` training rows

Because it is stochastic, it is a poor fit for deterministic reconstruction in [Trainer](Trainer.md).

## `xgboost_regressor`

`xgboost_regressor` is the regression-oriented foundational SFD.

Use this SFD for continuous targets rather than binary targets.

In the bundled smoke path, it prepared:

- `49` training features
- `3615` training rows

It requires `xgboost`.

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

## Running one immediately

A built-in YAML template can run directly through the CLI:

```bash
limen init built-in-logreg.yaml --template logreg_binary
limen validate built-in-logreg.yaml
limen profile built-in-logreg.yaml
limen run --dry-run built-in-logreg.yaml
limen run built-in-logreg.yaml
```

Direct Python use is still available when you need to integrate with UEL or custom code. When `data=` is omitted on a manifest-driven SFD, the manifest fetches data using `fetch_data()`.

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
