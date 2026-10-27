# `limen.features`

> Build higher-level model inputs and target helpers on top of raw bars and indicator columns.

## Canonical docs

- [Features](../../docs/Features.md)
- [Conserved Flux Renormalization](../../docs/Conserved-Flux-Renormalization.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)

## In the research projects

The [Python project](../../docs/examples/python-research/README.md) owns its feature preparation and retains diagnostics for every exported helper. The [Compiled project](../../docs/examples/compiled-research/README.md) declares lagged and fractional features, conditional transforms and individual helper groups in YAML. CFR separately uses recorded trade-level observations. The feature reference gives the required columns, timing and output meaning.

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

| Module or family | Responsibility |
|---|---|
| `lagged_features.py` | Lag helpers for arbitrary columns |
| `breakout_*.py` | Breakout and threshold features |
| `*_volatility.py` | Range-based and higher-order volatility features |
| `*liquidity*.py, dollar_volume.py` | Bar-derived liquidity and impact features |
| `*seasonality.py` | Hour-of-week normalized bar behavior |
| `*intensity.py, *_range.py` | Candle anatomy and auction-style structure |
| `volume_*.py, ma_slope_regime.py` | Regime and structure features |
| `*_time_features.py` | Calendar and cyclical time context |
| `distance_from_*.py, range_pct.py` | Position and range features |
| `vwap.py, ichimoku_cloud.py` | Higher-level technical composites |

<!-- reference: package-features-layout source=limen/features/README.md region=layout:package role=layout -->
```text
features/
├── README.md
├── __init__.py
├── _regime_audit.py
├── _regime_config.py
├── _regime_contract.py
├── _regime_ranks.py
├── _regime_rules.py
├── _regime_state.py
├── absorption_intensity.py
├── active_lines.py
├── active_quantile_count.py
├── amihud_illiquidity.py
├── atr_percent_sma.py
├── atr_sma.py
├── body_to_range.py
├── breakout_features.py
├── breakout_percentile_regime.py
├── bulk_volume_classification.py
├── calendar_time_features.py
├── classify_market_regime.py
├── close_ma_distance_atr.py
├── close_position.py
├── close_position_rolling.py
├── close_to_extremes.py
├── common_strategy_positions.py
├── conserved_flux_renormalization.py
├── cusum_filter.py
├── cyclical_time_features.py
├── distance_from_high.py
├── distance_from_low.py
├── distance_from_ma.py
├── dollar_bar_crash_reversal.py
├── dollar_volume.py
├── downside_volatility_ratio.py
├── dynamic_stop_loss.py
├── dynamic_target.py
├── ema_alignment.py
├── entry_score_microstructure.py
├── feature_aliases.py
├── fractional_diff.py
├── gap_high.py
├── garman_klass_volatility.py
├── hh_hl_structure_regime.py
├── hours_since_big_move.py
├── hours_since_quantile_line.py
├── ichimoku_cloud.py
├── illiquidity_shock.py
├── is_funding_hour.py
├── is_us_open_hour.py
├── jump_variation_proxy.py
├── kaufman_efficiency_ratio.py
├── kline_imbalance.py
├── lagged_features.py
├── liquidity_drop.py
├── liquidity_range.py
├── log_returns.py
├── ma_slope_regime.py
├── maker_liquidity_share.py
├── maker_volume_ratio.py
├── maker_volume_share.py
├── market_regime.py
├── micro_momentum.py
├── momentum_confirmation.py
├── momentum_periods.py
├── momentum_weight.py
├── narrow_range.py
├── order_flow_imbalance.py
├── parkinson_vol_of_vol.py
├── parkinson_volatility.py
├── position_in_candle.py
├── position_in_range.py
├── price_lines.py
├── price_range_position.py
├── price_vs_band_regime.py
├── quantile_line_density.py
├── quantile_price_lines.py
├── range_overlap.py
├── range_pct.py
├── range_per_dollar_volume.py
├── realized_kurtosis.py
├── realized_semivariance.py
├── realized_skewness.py
├── regime_multiplier.py
├── rejection_intensity.py
├── relative_range_seasonality.py
├── relative_volatility_seasonality.py
├── relative_volume_seasonality.py
├── return_autocorrelation.py
├── return_per_dollar_volume.py
├── return_volatility_correlation.py
├── returns_lags.py
├── rogers_satchell_volatility.py
├── rolling_zscore.py
├── sma_crossover.py
├── sma_ratios.py
├── spread.py
├── spread_percent.py
├── stochastic_k_abs.py
├── tail_event_intensity.py
├── taker_imbalance_ratio.py
├── time_to_funding.py
├── trade_density.py
├── trade_imbalance.py
├── trade_size_ratio.py
├── trend_coherence.py
├── trend_strength.py
├── volatility_1h.py
├── volatility_barriers.py
├── volatility_measure.py
├── volatility_of_volatility.py
├── volatility_ratio.py
├── volatility_spike.py
├── volatility_term_structure.py
├── volatility_weight.py
├── volume_ratio.py
├── volume_regime.py
├── volume_spike.py
├── volume_to_range.py
├── volume_trend.py
├── volume_volatility_correlation.py
├── volume_weight.py
├── vpin.py
├── vwap.py
├── wick_imbalance.py
├── wick_proportion.py
├── window_return_regime.py
└── yang_zhang_volatility.py
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
