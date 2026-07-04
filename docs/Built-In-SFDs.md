# Built-in SFDs

Limen ships foundational SFDs under `limen.sfd.foundational_sfd` and matching YAML templates under `limen/yaml/templates`. Ordinary runs should start from the YAML template and CLI; the Python modules are the packaged decoder layer beneath that path.

They show the packaged Limen experiment shape because each one combines:

- `params()`
- `manifest()`
- a matching reference-architecture model surface

## The current catalog

| SFD | Task shape | Notes |
|---|---|---|
| `logreg_binary` | binary classification | canonical manifest-driven logistic-regression reference flow |
| `lightgbm_binary` | binary classification | the tradeline long-binary experiment: line-geometry features, train-fitted breakout target, LightGBM classifier |
| `random_binary` | binary classification baseline | sanity-check and control-comparison flow |
| `xgboost_regressor` | regression | tree-based regression workflow |
| `tabpfn_binary` | binary classification | optional, available only when `tabpfn` is installed |

## Foundational SFD versus reference architecture

Each built-in SFD has a matching model module in [Reference Architecture](Reference-Architecture.md).

The split is:

| Layer | Owns |
|---|---|
| foundational SFD | search space plus manifest pipeline |
| reference architecture | class-based model contract and function wrapper |

For the logistic-regression SFD:

- `limen.sfd.foundational_sfd.logreg_binary` owns the packaged experiment
- `limen.sfd.reference_architecture.logreg_binary` owns the model implementation

This separation is what lets [Trainer](Trainer.md) reconstruct a finished experiment and retrain the matching `ReferenceModel`.

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

## `tabpfn_binary`

`tabpfn_binary` is an optional packaged SFD. It only becomes available when `tabpfn` is installed through the `tabpfn` extra. That dependency is intentionally outside the base install because it is materially larger than the default sklearn/LightGBM path.

It uses `CalibrationBuilder` with the same probability calibration and threshold optimisation wiring as `logreg_binary`, so its results also include `optimal_threshold` and `val_score` when calibration is active.

That optional status matters at import time and in local documentation examples. In the bundled smoke path, it was unavailable because `tabpfn` was not installed.

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
- Choose `tabpfn_binary` only when that dependency is installed and the TabPFN workflow is required.

## Read next

- Continue to [Single-File Decoder](Single-File-Decoder.md) for the general SFD contract.
- Continue to [Command Line Interface](Command-Line-Interface.md) for the YAML run loop.
- Continue to [Reference Architecture](Reference-Architecture.md) for the class-based model layer underneath these built-in decoders.
- Continue to [Experiment Manifest](Experiment-Manifest.md) to adapt one of these into a custom SFD.
