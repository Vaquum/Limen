# `limen.targets`

> Build supervised target columns from market data for manifest-driven and custom experiments.

## Canonical docs

- [Targets](../../docs/Targets.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md#target-configuration)
- [Transforms](../../docs/Transforms.md)

## What this package owns

Owns train-fitted and transform-time target builders, including binary classification targets, return targets, breakout targets, barrier targets, and identity passthrough targets.
Does **not** own feature engineering, model fitting, calibration, or execution strategy.

## Key entry points

| Entry point | Use it when | Notes |
| --- | --- | --- |
| `QuantileBinaryTarget` | You need labels from a train-fitted quantile cutoff | Fits the cutoff on train only. |
| `ThresholdBinaryTarget` | You need labels from a fixed threshold | Useful for explicit return thresholds. |
| `TradelineLongBinaryTarget` | You need a long-entry label from price-line context | Pairs with line-based feature helpers. |
| `NextReturnTarget`, `VolNormalizedReturnTarget`, `ForwardVolNormalizedReturnTarget` | You need continuous forward-return style targets | Common for regression or ranking experiments. |
| `NextBarUpTarget`, `NextBarDownTarget` | You need one-bar direction labels | Lightweight classification baselines. |
| `ForwardBreakoutTarget`, `EmaBreakoutTarget` | You need breakout-oriented labels | Uses forward path or EMA context. |
| `TripleBarrierTarget` | You need a path-aware barrier label | Requires enough rows for `max_horizon` and the volatility warmup. |
| `RiskRewardRatioTarget`, `ExitQualityTarget` | You need outcome labels from upstream trade-path columns | Require the expected input columns to exist on every split. |
| `RandomBinaryTarget`, `IdentityTarget` | You need controls or already-present target columns | Useful for tests, baselines, and externally labelled data. |

## Adjacent modules

- `limen.transforms` can derive target inputs before target construction.
- `limen.experiment.Manifest` wires target builders into `prepare_data()`.
- `limen.calibration` evaluates predictions against the target labels after model output.

## Quick orientation

```text
targets/
|-- quantile_binary.py
|-- threshold_binary.py
|-- next_bar_up.py / next_bar_down.py
|-- next_return.py
|-- vol_normalized_return.py / forward_vol_normalized_return.py
|-- forward_breakout.py / ema_breakout.py / tradeline_long_binary.py
|-- triple_barrier.py
|-- risk_reward_ratio.py / exit_quality.py
|-- identity.py
`-- random_binary.py
```

## Things to know

- Target classes receive the training split during construction and apply `transform()` to every split.
- Train-fitted targets must store only train-derived state before validation and test transformation.
- Forward-looking targets naturally create boundary nulls; manifest preparation drops null target rows downstream.
- Outcome-derived targets require their upstream columns to exist before target construction.

## Read next

- [Targets](../../docs/Targets.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md#target-configuration)
