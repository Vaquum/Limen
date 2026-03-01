# `limen.features`

> Compute higher-level market features from OHLCV bars as Polars lazy-frame transformations.

## Responsibilities

Each function in this module takes a `pl.LazyFrame` (or `pl.DataFrame`) plus optional parameters and returns a `pl.LazyFrame` with one or more new columns appended.  Features are composable pipeline steps, intended to be passed to `Manifest.add_feature()`.

Does **not** own raw indicator arithmetic (that is in `limen.indicators`) or target/label construction (that belongs in the SFD's model function or transforms).

## Key concepts

- **Feature function** – a pure, stateless function `(pl.LazyFrame, **params) → pl.LazyFrame`; each file exports one or a small family of related functions
- **Regime feature** – a categorical or integer column that partitions bars into market states (e.g. `market_regime`, `volume_regime`, `ma_slope_regime`)
- **Lagged feature** – shifts a column forward in time so the model sees past values; provided by `lagged_features.py`
- **Composite feature** – combines multiple indicators into a single derived signal (e.g. `conserved_flux_renormalization`, `entry_score_microstructure`)

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `lag_column()` / `lag_columns()` / `lag_range()` | `lagged_features.py` | Add time-lagged copies of one or more columns |
| `breakout_features()` | `breakout_features.py` | Add breakout distance / momentum columns |
| `ichimoku_cloud()` | `ichimoku_cloud.py` | Compute all five Ichimoku components |
| `vwap()` | `vwap.py` | Rolling VWAP relative to close |
| `kline_imbalance()` | `kline_imbalance.py` | Open/close body direction and imbalance ratio |
| `conserved_flux_renormalization()` | `conserved_flux_renormalization.py` | Multi-scale flux normalisation feature |
| `quantile_flag()` | `quantile_flag.py` | Binary flag: close above/below a rolling quantile |
| `market_regime()` | `market_regime.py` | Multi-factor regime label |
| `dynamic_stop_loss()` / `dynamic_target()` | `dynamic_stop_loss.py` / `dynamic_target.py` | ATR-based stop/target levels |

## Dependencies

- **Internal:** `limen.indicators` (several feature files call indicator functions)
- **External:** `polars`, `numpy`

## Quick orientation

```text
features/
├── lagged_features.py              # lag_column, lag_columns, lag_range, lag_range_cols
├── breakout_features.py            # breakout_features
├── conserved_flux_renormalization.py
├── ichimoku_cloud.py
├── vwap.py
├── kline_imbalance.py
├── quantile_flag.py
├── market_regime.py
├── dynamic_stop_loss.py / dynamic_target.py
├── *_regime.py                     # Various categorical regime signals
├── feature_aliases.py              # Shared column-name constants
└── … (one file per feature or family)
```

## Gotchas / things to know

- All functions operate on lazy frames to stay composable; call `.collect()` only at the end of the pipeline
- Parameter values that look like `_param_name` or that match a key in `round_params` are resolved at runtime by the `Manifest._resolve_params()` mechanism — do not hard-code param values in `Manifest.add_feature()` calls
- Lag functions introduce `null` rows at the head of the frame; `Manifest.prepare_data()` calls `drop_nulls()` after all transforms
