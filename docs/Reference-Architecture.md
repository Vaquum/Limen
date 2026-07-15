# Reference Architecture

The reference architecture is Limen's class-based model layer. It sits underneath the foundational SFDs and underneath `Trainer`.

This page covers:

- what a `ReferenceModel` must implement
- how the built-in model classes behave
- what `evaluate(data, inline_metrics=True)` adds
- how class-based models relate to the simpler function wrappers used in manifests

## Public Surface

The current public reference-architecture exports are:

- `ReferenceModel`
- `DLinearRegressor`
- `LightGBMBinary`
- `LogRegBinary`
- `RandomBinary`
- `RuleBasedStrategy`
- `XGBoostRegressor`
- `TabPFNBinary`

Each model module also exposes a function-style wrapper with the same behavioral surface used by foundational manifests. The TabPFN symbols are importable without the optional dependency; constructing or training `TabPFNBinary` requires `vaquum-limen[tabpfn]`.

The function exports are `dlinear_regressor`, `lightgbm_binary`, `logreg_binary`, `random_binary`, `rule_based`, `tabpfn_binary`, and `xgboost_regressor`.

## `ReferenceModel`

`ReferenceModel` is the base contract. Every subclass must implement:

```python
train(data, **params)
predict(data)
evaluate(data, inline_metrics=True)
```

### Data expectations

Manifest preparation can provide:

- `x_train`, `y_train`
- `x_val`, `y_val`
- `x_test`, `y_test`
- `price_data_for_backtest`
- `_feature_names`
- `_alignment`
- `_scaler`

Models consume the subset of keys they need from the standard Limen shape.

## The Built-In Model Classes

| Class | Task shape | Deterministic | Notes |
|---|---|---|---|
| `LogRegBinary` | binary classification | no | sklearn logistic regression wrapper; manifest wrapper exposes constructor params; solver refits are not bit-reproducible across BLAS builds and thread counts, so Trainer validates with relative tolerance |
| `LightGBMBinary` | binary classification | yes | LightGBM classifier exposing the full `LGBMClassifier` surface; accepts `binary`, `cross_entropy`, or `None` as its objective and rejects other objectives before training; early stopping on the validation split; reproducibility pinned via `deterministic`/`force_row_wise`/`random_state` defaults |
| `RandomBinary` | binary baseline | no | intentionally stochastic |
| `XGBoostRegressor` | regression | yes | requires `xgboost`; training is bit-reproducible for a fixed `random_state` on a fixed environment (verified byte-identical predictions across processes and thread counts), so Trainer validates with its near-exact deterministic tolerance |
| `DLinearRegressor` | regression | yes | canonical DLinear semantics; closed-form SVD ridge fit, no seed; requires `scipy` |
| `TabPFNBinary` | binary classification | no | optional, requires `tabpfn` |
| `RuleBasedStrategy` | rule-based long/flat | yes | no training step; boolean predicate logic |

The `deterministic` flag matters because [Trainer](Trainer.md) uses it to choose its validation tolerance.

## Probability Support for Cohort

For Cohort, “probability” always means **P(1)**: the probability that the positive class is `1`.

Architectures that expose valid P(1) may use Cohort's probability-weighted aggregation path. Architectures that do not expose valid P(1) use Cohort's majority-vote fallback path instead.

| Architecture | Returns probabilities P(1) | Cohort mode | Notes |
|---|---:|---|---|
| `LogRegBinary` | yes | probability | `predict()` returns `_probs = predict_proba(x_test)[:, 1]`, which is directly the class-1 probability P(1). |
| `LightGBMBinary` | yes | probability | `predict()` returns `_probs` from the classifier or fitted calibrator. |
| `RandomBinary` | yes | probability | `predict()` returns `_probs`, but they are synthetic confidence values (`0.9` for predicted 1, `0.1` for predicted 0), not model-derived calibrated probabilities. Still usable as P(1)-shaped output if Cohort accepts implementation-defined probability-like outputs. |
| `TabPFNBinary` | yes | probability | `predict()` returns `_probs` as positive-class probability. When a `CalibrationConfig` is configured, probabilities are optionally recalibrated and the threshold optimized before `_preds` are produced. This is compatible with P(1). |
| `XGBoostRegressor` | no | fallback | `predict()` returns only `_preds` and does not expose `_probs`. Since this is a regressor, any Cohort use would have to fall back unless a separate binary-probability wrapper is introduced. |
| `DLinearRegressor` | no | fallback | `predict()` returns only `_preds`. Same regressor caveat as `XGBoostRegressor`. |
| `RuleBasedStrategy` | no | fallback | `predict()` returns positions as `_preds`; `_probs` is intentionally absent. |

## `predict()` Versus `evaluate()`

`predict()` is the small inference surface. For the built-in binary models it always returns:

- `_preds`
- `_probs`

When a `CalibrationConfig` is configured on the manifest and injected into the architecture, `predict()` additionally returns:

- `optimal_threshold` — the threshold chosen by the optimizer (or `0.5` when only probability calibration is configured)
- `val_score` — the metric score at that threshold; `None` when no threshold function is set

`evaluate()` passes these keys through into the results dict alongside the standard binary metrics.

`evaluate()` is the richer offline evaluation surface.

### `inline_metrics=False`

With `inline_metrics=False`, `evaluate()` omits inline confusion and backtest metrics. The returned dictionary still includes the five task metrics and private or schema-stability fields required by the architecture.

For `LogRegBinary`, the task metrics are:

- `accuracy`
- `auc`
- `precision`
- `recall`
- `fpr`

`LogRegBinary` also returns `_preds`, `optimal_threshold`, and `val_score`; the last two are `None` when no calibration or threshold step produced them. Other architectures may retain their own private prediction payloads.

### `inline_metrics=True`

With `inline_metrics=True`, `evaluate()` adds:

- `confusion_*` metrics
- `confusion_*_mean_return_pct` and `backtest_*` metrics when `price_data_for_backtest` is present

When `price_data_for_backtest` is absent, default inline evaluation still returns task metrics and confusion counts; price-derived confusion-return and backtest metrics are skipped.

For `LogRegBinary`, inline evaluation can add keys such as:

- `backtest_edge_bps_p50`
- `backtest_pnl_bps_p50`
- `backtest_cvar_95_pnl_bps`
- `confusion_tp`
- `confusion_fp`
- `confusion_precision`

That is why the reference-architecture layer is the bridge between raw model output and the experiment-level analytics surfaces.

## Example: Class-Based Usage

```python
from limen.sfd.reference_architecture import LogRegBinary

model = LogRegBinary().train(
    data,
    solver='lbfgs',
    penalty='l2',
    C=0.1,
    class_weight=0.55,
    max_iter=60,
)

pred = model.predict({'x_test': data['x_test']})
results = model.evaluate(data, inline_metrics=True)
```

This is the same contract that `Trainer` replays before it wraps finished experiment rounds in `Sensor` objects.

## Function Wrappers Versus Classes

Foundational manifests call the function wrapper:

```python-fragment
.with_reference_architecture(logreg_binary)
```

That wrapper:

1. instantiates the matching class
2. trains it
3. evaluates it with `inline_metrics=True`

The class is the canonical reusable architecture surface. The function wrapper is the manifest-facing adapter.

## Trainer Relationship

`Trainer` reconstructs the compiled manifest, reruns `prepare_data()` and `run_model()`, and uses the returned `_model` after metric validation.

The class-based layer matters even when daily work touches foundational SFDs:

- foundational SFDs package the experiment
- reference architecture owns the model contract
- `Trainer` replays selected rounds into validated class-based models

Deterministic models use near-exact metric tolerance; stochastic models use a wider relative tolerance and can still raise `ReconstructionError` when a replay diverges too far.

## `DLinearRegressor`

`DLinearRegressor` is Limen's canonical DLinear surface (Zeng et al. 2023, *Are Transformers Effective for Time Series Forecasting?*). It owns the DLinear semantics that downstream parity checks compare against, so every semantic choice is explicit:

- **Decomposition** — each lookback window is split by a centered moving average with replicate edge padding (the official DLinear `moving_avg` behavior; `kernel_size` must be a positive odd integer, canonical default `25`) into a trend component and a remainder component.
- **Heads** — each component gets its own linear head and the head outputs are summed, which is exactly the DLinear function class.
- **Fit** — the ridge-regularized DLinear MSE objective is minimized in closed form by SVD (`alpha` is the ridge strength; `alpha=0` gives the min-norm least-squares solution). There is no optimizer loop and no seed: `deterministic = True`, and re-running the same config reproduces the experiment log exactly on a fixed environment. Cross-BLAS-build float differences remain possible, as with any LAPACK-backed path.
- **Lookback** — the window is read from feature columns named `ret_1_lag_{i}`, ordered by lag descending so each row is the window in time order (oldest bar first). Window length is a manifest concern (`lag_range` with `end=lookback_end`); all other feature columns are ignored.
- **Target shape** — scalar per bar: `NextReturnTarget(periods=horizon, scale=100.0)`, the percentage return over the next `horizon` bars.
- **Direction mapping** — `evaluate()` mirrors `XGBoostRegressor`: `preds > 0` becomes the long/flat signal for the inline confusion and backtest metrics.

`DLinearRegressor` is a reference baseline, not an alpha source. On the measured reference window (1h BTCUSDT spot klines from 2025-01-01, split 8:1:2, 5 bps fee + 5 bps slip per fill, 1-bar execution lag), gross `edge_bps_p50` was −0.3 to 0.0 bps, the best actively-trading config netted −1.4 bps/bar against −0.4 bps/bar for always-long, and out-of-sample r² was ≈ 0.000 — at parity with `XGBoostRegressor` on the same window. No sign-honest configuration cleared 10 bps/side costs at these frequencies.

## `RuleBasedStrategy`

`RuleBasedStrategy` is the reference architecture for rule-based SFDs. It differs from the ML models in two important ways:

- `train()` is a no-op — rule-based strategies have no learnable parameters
- `evaluate()` runs across all three splits (train/val/test) and returns cross-split stability metrics in addition to per-split backtest metrics

It expects a `data_dict` produced by a manifest configured with `with_strategy()`:

```python
{
    'train': pl.DataFrame,  # with pre-computed boolean predicate columns
    'val':   pl.DataFrame,
    'test':  pl.DataFrame,
    'strategy': {
        'conditions': [
            {'id': 'entry_id', 'type': 'threshold', 'column': 'wilder_rsi_14', 'operator': '<', 'value': 30},
        ],
        'entry': 'entry_id',
    },
}
```

The strategy walks the boolean logic tree defined in `strategy['conditions']`, resolves each leaf condition by reading its pre-computed boolean column from the split DataFrame, and folds compound conditions via AND/OR/NOT. Positions are 0 (flat) or 1 (long) per bar.

### Metrics produced

`evaluate()` returns a flat dict with three tiers:

- **Tier 1** — position stats: `num_trades_{split}`, `position_rate_{split}`
- **Tier 2** — per-split backtest: all `backtest_snapshot` output columns plus rule-only `pnl_per_trade_bps` and `num_executed_trades`, each suffixed with `_{split}` (e.g. `pnl_bps_p50_train`, `drawdown_bps_p5_test`, `pnl_per_trade_bps_test`)
- **Tier 3** — cross-split diagnostics: `drawdown_std_bps`, `is_stable`

`is_stable` is `False` until a replacement stability rule is defined against the new decoder-level ledger.

`pnl_per_trade_bps` is intentionally outside the generic 20-column snapshot. `RuleBasedStrategy` derives it from contiguous executed-position segments returned by the shared long-flat strategy, reports their count as `num_executed_trades`, compounds net per-bar returns after costs and notional sizing within each segment, then takes the arithmetic mean. No executed trades yields a zero count and `NaN` mean.

**NOTE:** Tier 2 and Tier 3 metrics require `open` and `close` columns in the split DataFrames to run the backtest. When those columns are absent, per-split backtest results are empty and `drawdown_std_bps` is returned as `None` with `is_stable` falling back to `False`.

`_preds` is present; `_probs` is intentionally absent (not applicable to rule-based strategies).

## Optional Dependencies

- `xgboost_regressor` requires `xgboost`
- `dlinear_regressor` requires `scipy` (the `stats` extra), loaded lazily inside the model
- `tabpfn_binary` requires `tabpfn`

Optional dependencies are checked when the relevant model executes, not when the package is imported.

## Read Next

- Continue to [Built-In SFDs](Built-In-SFDs.md) to see how the shipped foundational SFDs package these model surfaces.
- Continue to [Trainer](Trainer.md) for the reconstruction workflow that replays and validates selected rounds.
- Continue to [Standard Metrics Library](Standard-Metrics-Library.md) for the low-level metric helpers used inside these model classes.
