# Built-In SFDs

Limen ships a small set of foundational SFDs under `limen.sfd.foundational_sfd`. These are the packaged decoders you can run immediately without authoring your own experiment module first.

They are the fastest way to learn how Limen is shaped in practice because each one already combines:

- `params()`
- `manifest()`
- a matching reference-architecture model surface

## The Current Catalog

| SFD | Task shape | Notes |
|---|---|---|
| `logreg_binary` | binary classification | the main manifest-driven logistic-regression reference flow |
| `random_binary` | binary classification baseline | useful for sanity checks and control comparisons |
| `xgboost_regressor` | regression | tree-based regression workflow |
| `tabpfn_binary` | binary classification | optional, available only when `tabpfn` is installed |

## Foundational SFD Versus Reference Architecture

Each built-in SFD has a matching model module in [Reference Architecture](Reference-Architecture.md).

The split is:

| Layer | Owns |
|---|---|
| foundational SFD | search space plus manifest pipeline |
| reference architecture | class-based model contract and function wrapper |

So, for example:

- `limen.sfd.foundational_sfd.logreg_binary` owns the packaged experiment
- `limen.sfd.reference_architecture.logreg_binary` owns the model implementation

This separation is what lets [Trainer](Trainer.md) reconstruct a finished experiment and retrain the matching `ReferenceModel`.

## `logreg_binary`

`logreg_binary` is the standard manifest-driven binary classifier in the package.

It currently combines:

- indicators such as `roc`, `atr`, `ppo`, and `wilder_rsi`
- features such as `vwap` and `kline_imbalance`
- a fitted quantile-based target
- scaler selection from params (`logreg`, `robust`, `rank_gauss`)
- the `LogRegBinary` reference model
- `CalibrationBuilder` with `sklearn_probability_calibrator` and `grid_threshold_optimizer`

The classifier parameter surface mirrors the sklearn `LogisticRegression` constructor through manifest params: `solver`, `penalty`, `dual`, `tol`, `C`, `fit_intercept`, `intercept_scaling`, `class_weight`, `random_state`, `max_iter`, `multi_class`, `verbose`, `warm_start`, `n_jobs`, and `l1_ratio`.

The calibration search space includes `use_calibration`, `use_threshold`, `cal_method`, `threshold_min`, `threshold_max`, and `threshold_step`, giving a full grid of calibration modes within a single experiment run.

On a live local smoke run over the bundled test dataset in this repo, it prepared:

- `24` training features
- `3610` training rows

## `random_binary`

`random_binary` is the baseline binary classifier. It is deliberately simple and deliberately stochastic.

Use it when you want:

- a control run
- a smoke-test decoder
- a deliberately weak comparison point

On a live local smoke run in this repo, it prepared:

- `18` training features
- `2999` training rows

Because it is stochastic, it is a poor fit for deterministic reconstruction in [Trainer](Trainer.md).

## `xgboost_regressor`

`xgboost_regressor` is the regression-oriented foundational SFD.

Use it when the target is better treated as continuous rather than binary.

On a live local smoke run in this repo, it prepared:

- `49` training features
- `3615` training rows

It requires `xgboost`.

## `tabpfn_binary`

`tabpfn_binary` is an optional packaged SFD. It only becomes available when `tabpfn` is installed.

It uses `CalibrationBuilder` with the same probability calibration and threshold optimisation wiring as `logreg_binary`, so its results also include `optimal_threshold` and `val_score` when calibration is active.

That optional status matters at import time and in local documentation examples. In a live local smoke pass in this repo, it was unavailable because `tabpfn` was not installed.

## Running One Immediately

The simplest way to use a built-in SFD is:

```python
import limen

uel = limen.UniversalExperimentLoop(
    sfd=limen.sfd.logreg_binary,
)

uel.run(
    experiment_name='built-in-logreg',
    n_permutations=5,
    prep_each_round=True,
)
```

If you omit `data=`, the manifest fetches data using `fetch_data()`. Pass `test_mode=True` to UEL to use the test data source instead.

## How To Choose

- Choose `logreg_binary` when you want the clearest canonical Limen path.
- Choose `random_binary` when you want a baseline or smoke-test decoder.
- Choose `xgboost_regressor` when the target is continuous and tree-based regression is the better fit.
- Choose `tabpfn_binary` only when that dependency is installed and you specifically want the TabPFN workflow.

## Read Next

- Continue to [Single-File Decoder](Single-File-Decoder.md) for the general SFD contract.
- Continue to [Reference Architecture](Reference-Architecture.md) for the class-based model layer underneath these built-in decoders.
- Continue to [Experiment Manifest](Experiment-Manifest.md) if you want to adapt one of these into your own custom SFD.
