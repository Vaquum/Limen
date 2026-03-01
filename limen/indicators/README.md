# `limen.indicators`

> Compute standard technical indicators from OHLCV bars as Polars lazy-frame transformations.

## Responsibilities

Each function accepts a `pl.LazyFrame` and configuration parameters and returns a `pl.LazyFrame` with the indicator column(s) appended.  Indicators are the lowest-level numeric building blocks, suitable for direct use in `Manifest.add_indicator()` or as inputs to `limen.features` functions.

Does **not** own higher-level derived features, regime labels, or target construction.

## Key concepts

- **Indicator function** – a pure stateless function `(pl.LazyFrame, **params) → pl.LazyFrame` that appends one or more columns
- **Period-parameterised indicator** – most indicators accept a `period` (or `fast`/`slow`) argument so the experiment loop can search over window sizes
- **Rolling statistic** – computed with Polars `.rolling_*` expressions; windows shorter than the lookback produce `null` that `drop_nulls()` removes downstream

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `sma()` | `sma.py` | Simple moving average |
| `atr()` | `atr.py` | Average true range |
| `wilder_rsi()` | `wilder_rsi.py` | Wilder-smoothed RSI |
| `rsi_sma()` | `rsi_sma.py` | SMA-based RSI approximation |
| `macd()` | `macd.py` | MACD line and signal |
| `ppo()` | `ppo.py` | Percentage price oscillator |
| `bollinger_bands()` | `bollinger_bands.py` | Upper/lower Bollinger bands |
| `bollinger_position()` | `bollinger_position.py` | Close position within the Bollinger band |
| `cci()` | `cci.py` | Commodity channel index |
| `stochastic_oscillator()` | `stochastic_oscillator.py` | %K and %D stochastic |
| `roc()` | `roc.py` | Rate of change |
| `returns()` | `returns.py` | Per-bar log or simple returns |
| `rolling_volatility()` | `rolling_volatility.py` | Rolling standard deviation of returns |
| `window_return()` | `window_return.py` | Cumulative return over a sliding window |
| `body_pct()` | `body_pct.py` | Candle body as a fraction of the total range |
| `price_change_pct()` | `price_change_pct.py` | Percentage close-to-close change |
| `sma_deviation_std()` | `sma_deviation_std.py` | Distance from SMA in standard-deviation units |

## Dependencies

- **Internal:** none
- **External:** `polars`

## Quick orientation

```text
indicators/
├── sma.py                   # Simple moving average
├── atr.py                   # Average true range
├── wilder_rsi.py            # Wilder RSI
├── rsi_sma.py               # SMA-based RSI
├── macd.py                  # MACD
├── ppo.py                   # PPO
├── bollinger_bands.py       # Bollinger bands
├── bollinger_position.py    # Close position in band
├── cci.py                   # CCI
├── stochastic_oscillator.py # Stochastic %K/%D
├── roc.py                   # Rate of change
├── returns.py               # Per-bar returns
├── rolling_volatility.py    # Volatility
├── window_return.py         # Windowed cumulative return
├── body_pct.py              # Candle body ratio
├── price_change_pct.py      # Close-to-close %
└── sma_deviation_std.py     # SMA deviation in σ
```

## Gotchas / things to know

- All functions operate on lazy frames; collect only at the end of the transform pipeline
- Rolling windows shorter than `period` produce `null`; these rows are dropped by `Manifest.prepare_data()` after all transforms are applied
- Column names are deterministic (e.g. `atr_14`, `wilder_rsi_14`) so downstream features and scalers can reference them by pattern
