# `limen.indicators`

> Provide the lower-level technical indicator library that manifests and custom prep functions build on.

## Canonical docs

- [Indicators](../../docs/Indicators.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)

## What this package owns

Owns the public indicator function library, including moving averages, oscillators, volatility measures, price transforms, and candlestick-pattern helpers.
Does **not** own higher-level feature engineering, target creation, or model training.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `limen.indicators.*` exports | You want indicators in a manifest or custom prep pipeline | The package root re-exports the public indicator surface |
| `sma`, `ema`, `wilder_rsi`, `atr` | You need common directional and volatility baselines | Frequently used in foundational SFDs |
| `bbands`, `bollinger_bands`, `bollinger_position` | You want band-based regime or position signals | Useful before higher-level feature construction |
| `macd`, `ppo`, `roc`, `stoch`, `stochrsi` | You need momentum and oscillator families | Most emit multiple columns or parameterized output names |
| Candlestick helpers like `cdldoji` | You want pattern flags aligned with TA-Lib semantics | Exported at the package root alongside the rest of the library |

## Adjacent modules

- `limen.features` typically consumes indicator columns and turns them into more opinionated model inputs.
- `limen.experiment.Manifest` wires indicators into the prep pipeline through `.add_indicator(...)`.
- `limen.data` supplies the OHLCV frames that most indicators expect.

## Quick orientation

```text
indicators/
├── ma, sma, ema, wma, tema, trima, t3 ...   # Moving averages and trend baselines
├── rsi, wilder_rsi, cmo, willr, ultosc ...  # Momentum and oscillator families
├── atr, natr, trange, stddev, var ...       # Volatility and range measures
├── bbands, bollinger_bands, midpoint ...    # Band and price-position helpers
├── macd, macdfix, macdext, ppo ...          # Multi-column momentum helpers
└── cdl*.py                                  # Candlestick pattern detections
```

## Things to know

- Most indicator functions accept a `pl.LazyFrame` and return that frame with one or more appended columns.
- Leading `null` rows are expected for rolling calculations. Manifest-driven prep drops them after the feature and indicator stage.
- The public reference should be treated as the canonical list of exported helpers and output-column conventions.
- Many indicators aim for TA-Lib parity, so naming and behavior are intentionally close to that surface where applicable.

## Read next

- [Indicators](../../docs/Indicators.md)
- [Features](../../docs/Features.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
