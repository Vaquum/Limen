# `limen.features`

> Compute derived, higher-level features from OHLCV bar data for use as model inputs.

## Responsibilities

Owns the library of feature-engineering functions that transform raw or indicator-enriched bar DataFrames into semantically meaningful columns (regime flags, momentum composites, structural patterns, etc.).
Does **not** own raw technical indicators (those live in `limen.indicators`) or model training — feature functions are pure Polars `LazyFrame → LazyFrame` (or `DataFrame → DataFrame`) transforms.

## Key concepts

- **Feature function** – a callable that accepts a `pl.LazyFrame` (or `pl.DataFrame`) plus keyword parameters and returns an enriched frame with one or more new columns appended.
- **Regime feature** – binary or ordinal column that classifies the current market state (e.g. `volume_regime`, `ma_slope_regime`, `market_regime`).
- **Lagged feature** – a past value of any column shifted by N bars, produced by `lag_column` / `lag_columns` / `lag_range`.
- **Conserved flux renormalisation (CFR)** - computes multi-scale flux and entropy diagnostics from trade data and joins them back onto bar-level output.
- **Quantile flag** – binary label derived by comparing a value against a rolling or global quantile cutoff; used as the primary target in many SFDs.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| All public feature functions | `__init__.py` | Add to a `Manifest` via `.add_feature(func, **params)` or call directly in a custom `prep` |
| `quantile_flag()` | `quantile_flag.py` | Target-generation step in manifest `with_target()` block |
| `compute_quantile_cutoff()` | `quantile_flag.py` | `fit_param` step that computes the cutoff on train data only |
| `lag_column()` / `lag_range()` | `lagged_features.py` | Add time-lagged versions of any column |

## Dependencies

- **Internal:** `limen.indicators` is typically called first in the prep pipeline; feature functions may consume indicator columns
- **External:** `polars`

## Quick orientation
```text
features/
├── lagged_features.py           # lag_column, lag_columns, lag_range, lag_range_cols
├── quantile_flag.py             # quantile_flag, compute_quantile_cutoff
├── conserved_flux_renormalization.py
├── market_regime.py             # Additional internal regime helper
├── volume_regime.py / volume_*.py
├── momentum_*.py                # Various momentum composites
├── breakout_*.py                # Breakout detection features
├── distance_from_high.py / distance_from_low.py
├── ichimoku_cloud.py
├── vwap.py
├── sma_crossover.py / ma_slope_regime.py
├── feature_aliases.py           # Internal alias helper for composite feature sets
└── ...                          # ~60 feature files total
```

## Gotchas / things to know

- Feature functions operate on `LazyFrame` when called through the `Manifest` pipeline; prefer lazy-compatible expressions.
- `compute_quantile_cutoff` must be used as a `fit_param` so the cutoff is computed on **training** data only and then reused on val/test splits.
- The package contains additional helper files beyond the default `limen.features` export surface; `market_regime.py` and `feature_aliases.py` are examples of internal helpers that are not re-exported through `__init__.py`.
- Features that depend on indicator columns (e.g. `atr_sma`) implicitly require those indicators to be added to the manifest first.
