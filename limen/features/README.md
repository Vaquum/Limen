# `limen.features`

> Build higher-level model inputs and target helpers on top of raw bars and indicator columns.

## Canonical docs

- [Features](../../docs/Features.md)
- [Conserved Flux Renormalization](../../docs/Conserved-Flux-Renormalization.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)

## What this package owns

Owns the public feature-engineering layer, including regime features, lag helpers, breakout features, target helpers, and trade-shape features like CFR.
Does **not** own raw technical indicators, feature scaling, or model fitting.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `limen.features.*` exports | You want feature functions inside a manifest or custom prep pipeline | The package root re-exports the public feature surface |
| `quantile_flag` | You want a trainable binary target based on a cutoff | Common target helper in manifest-driven SFDs |
| `compute_quantile_cutoff` | You need the train-only fit parameter that powers `quantile_flag` | Designed to be used through `fit_param` |
| `lag_column`, `lag_columns`, `lag_range`, `lag_range_cols` | You want lagged versions of existing columns | Useful for both raw and derived features |
| `conserved_flux_renormalization` | You want trade-derived multi-scale flux diagnostics | Requires trade-level input rather than plain OHLCV bars |

## Adjacent modules

- `limen.indicators` usually runs first and produces columns that many features depend on.
- `limen.transforms` is commonly used alongside feature generation when building targets.
- `limen.experiment.Manifest` wires features into the prep pipeline through `.add_feature(...)`.

## Quick orientation

```text
features/
├── quantile_flag.py                 # Target helper + train-only cutoff helper
├── lagged_features.py               # Lag helpers for arbitrary columns
├── breakout_*.py                    # Breakout and threshold features
├── volume_*.py, ma_slope_regime.py  # Regime and structure features
├── distance_from_*.py, range_pct.py # Position and range features
├── vwap.py, ichimoku_cloud.py       # Higher-level technical composites
└── conserved_flux_renormalization.py
```

## Things to know

- The package root exports the public feature surface, but not every helper file in the directory is part of that surface.
- Most manifest-driven uses run features lazily on `pl.LazyFrame`, so lazy-friendly expressions are the safe default.
- Train-only target helpers must stay train-only. `compute_quantile_cutoff` is a good example of that pattern.
- Some features assume earlier indicator columns already exist. The public reference calls these dependencies out.

## Read next

- [Features](../../docs/Features.md)
- [Conserved Flux Renormalization](../../docs/Conserved-Flux-Renormalization.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
