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

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `limen.features.*` exports | Feature functions inside a manifest or custom prep pipeline | The package root re-exports the public feature surface |
| `lag_column`, `lag_columns`, `lag_range`, `lag_range_cols` | Lagged versions of existing columns | Works for raw and derived features |
| `calendar_time_features`, `cyclical_time_features` | Calendar context or cyclical encodings from `datetime` | Time-of-bar features for schedules, regimes, and model inputs |
| `parkinson_volatility`, `garman_klass_volatility`, `rogers_satchell_volatility`, `yang_zhang_volatility` | Range-based volatility estimates from OHLC bars | Captures range information beyond close-to-close volatility |
| `dollar_volume`, `amihud_illiquidity`, `return_per_dollar_volume`, `range_per_dollar_volume`, `illiquidity_shock` | Liquidity and impact proxies from OHLCV data | Keeps bar-derived liquidity features separate from trade-level microstructure |
| `maker_*`, `trade_*`, `liquidity_*`, `taker_imbalance_ratio` | Maker/taker and LOB-liquidity features emitted by native klines | Uses columns from `get_spot_klines` and `get_spot_dollar_klines` |
| `rolling_zscore` | One primitive for rolling z-score features | Supports `identity`, `log1p`, and `abs` transforms |
| `wick_proportion`, `stochastic_k_abs`, `distance_from_ma`, `close_ma_distance_atr`, `kaufman_efficiency_ratio` | Structural rolling OHLCV features | Adds bar-structure, distance, and path-efficiency context |
| `realized_semivariance`, `realized_skewness`, `realized_kurtosis`, `jump_variation_proxy`, `tail_event_intensity`, `volatility_of_volatility` | Asymmetry, jump, and tail context near recent returns | Risk-state features beyond volatility levels |
| `relative_volume_seasonality`, `relative_range_seasonality`, `relative_volatility_seasonality` | Current bar behavior normalized against hour-of-week baselines | Detects activity and range deviations in a 24/7 market |
| `body_to_range`, `wick_imbalance`, `range_overlap`, `rejection_intensity`, `absorption_intensity` | Candle anatomy and auction-style context from plain bars | Focuses on rejection, overlap, and body-versus-wick structure |
| `trend_coherence`, `volatility_term_structure` | Short/medium/long horizon agreement features | Summarizes cross-timescale alignment without a large feature bundle |
| `conserved_flux_renormalization` | Trade-derived multi-scale flux diagnostics | Requires trade-level input rather than plain OHLCV bars |

## Adjacent modules

- `limen.indicators` runs first when features depend on indicator columns.
- `limen.transforms` is commonly used alongside feature generation when building targets.
- `limen.experiment.Manifest` wires features into the prep pipeline through `.add_feature(vwap)`.

## Quick orientation

```text
features/
├── lagged_features.py               # Lag helpers for arbitrary columns
├── breakout_*.py                    # Breakout and threshold features
├── *_volatility.py                  # Range-based and higher-order volatility features
├── *liquidity*.py, dollar_volume.py # Bar-derived liquidity and impact features
├── *seasonality.py                  # Hour-of-week normalized bar behavior
├── *intensity.py, *_range.py        # Candle anatomy and auction-style structure
├── volume_*.py, ma_slope_regime.py  # Regime and structure features
├── *_time_features.py               # Calendar and cyclical time context
├── distance_from_*.py, range_pct.py # Position and range features
├── vwap.py, ichimoku_cloud.py       # Higher-level technical composites
└── conserved_flux_renormalization.py
```

## Things to know

- The package root exports the public feature surface, but not every helper file in the directory is part of that surface.
- Most manifest-driven uses run features lazily on `pl.LazyFrame`, so lazy-friendly expressions are the safe default.
- Some features assume earlier indicator columns already exist. The public reference calls these dependencies out.
- OHLCV-native feature helpers assume bars are already in chronological order before rolling or trailing calculations are applied.

## Read next

- [Features](../../docs/Features.md)
- [Conserved Flux Renormalization](../../docs/Conserved-Flux-Renormalization.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
