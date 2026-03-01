# `limen.indicators`

> Compute standard technical analysis indicators from OHLCV bar data.

## Responsibilities

Owns the library of technical indicator functions (moving averages, oscillators, volatility measures, momentum).
Does **not** own higher-level feature engineering (that lives in `limen.features`) or model training — indicator functions are stateless transforms from a bar DataFrame to an enriched DataFrame.

## Key concepts

- **Indicator function** – a callable that accepts a `pl.LazyFrame` and returns it with one or more new indicator columns appended; all parameters are passed as keyword arguments.
- **Period** – the lookback window used by rolling computations (e.g. RSI period, ATR period); typically a search parameter in the SFD `params()` dict.
- **Wilder smoothing** – used by `wilder_rsi` and `atr` for exponentially weighted rolling calculations matching the original Wilder definitions.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| All public indicator functions | `__init__.py` | Add to a `Manifest` via `.add_indicator(func, **params)` or call directly in a custom `prep` |
| `atr()` | `atr.py` | Average True Range — commonly used as a volatility baseline |
| `wilder_rsi()` | `wilder_rsi.py` | Wilder RSI — used in most foundational SFDs |
| `sma()` | `sma.py` | Simple Moving Average |
| `macd()` | `macd.py` | MACD line and signal |
| `bollinger_bands()` | `bollinger_bands.py` | Upper/lower bands + bandwidth |
| `ppo()` | `ppo.py` | Percentage Price Oscillator |

## Dependencies

- **Internal:** none — this is a leaf module
- **External:** `polars`

## Quick orientation
```text
indicators/
├── atr.py                   # Average True Range (Wilder method)
├── body_pct.py              # Candle body as % of total range
├── bollinger_bands.py       # Bollinger Bands (upper, lower, bandwidth)
├── bollinger_position.py    # Price position within Bollinger Bands
├── cci.py                   # Commodity Channel Index
├── macd.py                  # MACD line and signal line
├── midpoint.py              # Midpoint Over Period (TA-Lib MIDPOINT)
├── ppo.py                   # Percentage Price Oscillator
├── price_change_pct.py      # Bar-over-bar percentage price change
├── returns.py               # Log or simple returns
├── roc.py                   # Rate of Change
├── rolling_volatility.py    # Rolling standard deviation of returns
├── rsi_sma.py               # RSI smoothed with SMA
├── sma.py                   # Simple Moving Average
├── sma_deviation_std.py     # Deviation from SMA in std units
├── stochastic_oscillator.py # Stochastic %K and %D
├── wilder_rsi.py            # RSI using Wilder smoothing
└── window_return.py         # Return over a fixed future/past window
```

## Gotchas / things to know

- Indicators produce `null` values in the first `period - 1` rows; the experiment pipeline calls `drop_nulls()` after feature/indicator transforms, so leading nulls are automatically removed.
- `ppo` defaults use short/long periods of 12/26; override via keyword arguments in the manifest.
- `window_return` can compute both forward-looking (target) and backward-looking (feature) returns depending on the sign of the `shift` parameter.
