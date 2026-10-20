# `limen.indicators`

> Provide the lower-level technical indicator library that manifests and custom prep functions build on.

## Canonical docs

- [Indicators](../../docs/Indicators.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)

## In the research projects

The [Python project](../../docs/examples/python-research/README.md) owns its decimal momentum expression. The [Compiled project](../../docs/examples/compiled-research/README.md) declares the package’s percent-valued `roc`. The indicator reference develops these two preparation choices and keeps their units and warm-up behavior explicit.

## What this package owns

Owns the public indicator function library, including moving averages, oscillators, volatility measures, price transforms, and candlestick-pattern helpers.
Does **not** own higher-level feature engineering, target creation, or model training.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `limen.indicators.*` exports | Indicators in a manifest or custom prep pipeline | The package root re-exports the public indicator surface |
| `sma`, `ema`, `wilder_rsi`, `atr` | Common directional and volatility baselines | Frequently used in foundational SFDs |
| `bbands`, `bollinger_bands`, `bollinger_position` | Band-based regime or position signals | Inputs for higher-level feature construction |
| `macd`, `ppo`, `roc`, `stoch`, `stochrsi` | Momentum and oscillator families | Most emit multiple columns or parameterized output names |
| Candlestick helpers like `cdldoji` | Pattern flags aligned with TA-Lib semantics | Exported at the package root alongside the rest of the library |

## Adjacent modules

- `limen.features` consumes indicator columns and turns them into more opinionated model inputs.
- `limen.experiment.Manifest` wires indicators into the prep pipeline through `.add_indicator(roc)`.
- `limen.data` supplies the OHLCV frames that most indicators expect.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `ma, sma, ema, wma, tema, trima, t3` | Moving averages and trend baselines |
| `rsi, wilder_rsi, cmo, willr, ultosc` | Momentum and oscillator families |
| `atr, natr, trange, stddev, var` | Volatility and range measures |
| `bbands, bollinger_bands, midpoint` | Band and price-position helpers |
| `macd, macdfix, macdext, ppo` | Multi-column momentum helpers |
| `cdl*.py` | Candlestick pattern detections |

<!-- reference: package-indicators-layout source=limen/indicators/README.md region=layout:package role=layout -->
```text
indicators/
├── README.md
├── __init__.py
├── _atr.py
├── _ema.py
├── _hilbert.py
├── ad.py
├── adosc.py
├── apo.py
├── atr.py
├── avgprice.py
├── bbands.py
├── body_pct.py
├── bollinger_bands.py
├── bollinger_position.py
├── bop.py
├── cci.py
├── cdl2crows.py
├── cdl3blackcrows.py
├── cdl3inside.py
├── cdl3linestrike.py
├── cdl3starsinsouth.py
├── cdl3whitesoldiers.py
├── cdlabandonedbaby.py
├── cdladvancedblock.py
├── cdlbelthold.py
├── cdlclosingmarubozu.py
├── cdlconcealbabyswall.py
├── cdlcounterattack.py
├── cdldarkcloudcover.py
├── cdldoji.py
├── cdldragonflydoji.py
├── cdlengulfing.py
├── cdlgravestonedoji.py
├── cdlhammer.py
├── cdlhangingman.py
├── cdlharami.py
├── cdlharamicross.py
├── cdlhighwave.py
├── cdlhikkake.py
├── cdlhikkakemod.py
├── cdlhomingpigeon.py
├── cdlidentical3crows.py
├── cdlinvertedhammer.py
├── cdlladderbottom.py
├── cdllongleggeddoji.py
├── cdllongline.py
├── cdlmarubozu.py
├── cdlmatchinglow.py
├── cdlmathold.py
├── cdlonneck.py
├── cdlpiercing.py
├── cdlrickshawman.py
├── cdlrisefall3methods.py
├── cdlseparatinglines.py
├── cdlshootingstar.py
├── cdlshortline.py
├── cdlspinningtop.py
├── cdlstalledpattern.py
├── cdlsticksandwich.py
├── cdltakuri.py
├── cdlthrusting.py
├── cdltristar.py
├── cdlunique3river.py
├── cmo.py
├── dema.py
├── ema.py
├── ht_dcperiod.py
├── ht_dcphase.py
├── ht_phasor.py
├── ht_sine.py
├── ht_trendline.py
├── ht_trendmode.py
├── kama.py
├── linearreg.py
├── linearreg_angle.py
├── linearreg_intercept.py
├── linearreg_slope.py
├── ma.py
├── macd.py
├── macdext.py
├── macdfix.py
├── mama.py
├── medprice.py
├── mfi.py
├── midpoint.py
├── midprice.py
├── mom.py
├── natr.py
├── obv.py
├── ppo.py
├── price_change_pct.py
├── returns.py
├── roc.py
├── rocp.py
├── rocr.py
├── rocr100.py
├── rolling_volatility.py
├── rsi.py
├── rsi_sma.py
├── sar.py
├── sarext.py
├── sma.py
├── sma_deviation_std.py
├── stddev.py
├── stoch.py
├── stochastic_oscillator.py
├── stochf.py
├── stochrsi.py
├── t3.py
├── tema.py
├── trange.py
├── trima.py
├── trix.py
├── tsf.py
├── typprice.py
├── ultosc.py
├── var.py
├── wclprice.py
├── wilder_rsi.py
├── willr.py
├── window_return.py
└── wma.py
```

## Things to know

- Indicator functions accept a `pl.LazyFrame` and return that frame with one or more appended columns.
- Leading `null` rows are expected for rolling calculations. Manifest-driven prep drops them after the feature and indicator stage.
- The public reference should be treated as the canonical list of exported helpers and output-column conventions.
- TA-Lib-aligned indicators keep naming and behavior close to that surface where applicable.

## Read next

- [Indicators](../../docs/Indicators.md)
- [Features](../../docs/Features.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
