# Indicators

Indicators are Limen's lowest-level signal helpers. They usually take a kline-style `pl.DataFrame`, append one or more derived columns, and return the same frame shape back for the rest of the manifest pipeline.

Use this page when you need to choose an indicator, understand its naming convention, or see which columns it adds by default. For higher-level derived signals, see [Features](Features.md).

## Conventions

- Indicators append columns. They do not replace the input frame with a reduced schema.
- Most helpers assume the usual OHLCV names: `open`, `high`, `low`, `close`, `volume`.
- TA-Lib-backed wrappers treat TA-Lib parity as the source of truth. If Limen and TA-Lib diverge, TA-Lib wins unless there is a deliberate, documented reason not to.
- Output column names usually encode the main defaults, such as `rsi_14` or `apo_12_26_0`.
- Candlestick pattern helpers add a same-named integer column, usually following TA-Lib's positive bullish, negative bearish, zero neutral convention.
- A few helpers intentionally expose legacy compatibility behavior. The main example is `sma`, which adds both `sma_{period}` and `{price_col}_sma_{period}`.

## Quick Example

```python
from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.indicators import atr, ppo, wilder_rsi
from limen.sfd.reference_architecture import logreg_binary


def manifest():
    return (
        Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
        )
        .set_test_data_source(method=HistoricalData._get_data_for_test)
        .set_split_config(8, 1, 2)
        .add_indicator(atr, period=14)
        .add_indicator(ppo)
        .add_indicator(wilder_rsi)
        .with_model(logreg_binary)
    )
```

## Family Index

- `Price and moving averages`: direct price transforms and smoothing helpers.
- `Momentum and oscillators`: rate-of-change, oscillator, regression, and Hilbert-style cycle helpers.
- `Volatility, bands, and range`: dispersion, band, and stop-style helpers.
- `Volume, breadth, and bar diagnostics`: flow, bar-body, and return-style helpers.
- `Candlestick patterns`: TA-Lib-style pattern flags over OHLC bars.

## Price And Moving Averages

These helpers usually take either a single `price_col` or basic OHLC inputs and add smoothed or transformed price views.

| Function | Adds by default | Notes |
|---|---|---|
| `avgprice` | `avgprice` | Average OHLC price. |
| `medprice` | `medprice` | Median of high and low. |
| `midprice` | `midprice_14` | Rolling midpoint of high and low over `period`. |
| `typprice` | `typprice` | Typical price from high, low, close. |
| `wclprice` | `wclprice` | Weighted close price. |
| `midpoint` | `midpoint_14` | Rolling midpoint of a single `price_col`. |
| `sma` | `sma_30`, `close_sma_30` | Adds both the canonical and compatibility alias columns. |
| `ema` | `ema_30` | Exponential moving average. |
| `dema` | `dema_30` | Double EMA. |
| `tema` | `tema_30` | Triple EMA. |
| `trima` | `trima_30` | Triangular moving average. |
| `t3` | `t3_5_0.7` | T3 moving average with `period` and `vfactor`. |
| `kama` | `kama_30` | Kaufman adaptive moving average. |
| `mama` | `mama`, `fama` | MESA adaptive moving average pair. |
| `ma` | `ma_30_0` | Generic moving average with explicit `ma_type`. |
| `ht_trendline` | `ht_trendline` | Hilbert Transform instantaneous trendline. |

## Momentum And Oscillators

These helpers focus on rate of change, directional persistence, oscillator state, or cycle structure.

| Function | Adds by default | Notes |
|---|---|---|
| `apo` | `apo_12_26_0` | Absolute Price Oscillator. |
| `ppo` | `ppo_12_26_0` | Percentage Price Oscillator. |
| `macd` | `macd`, `macd_signal`, `macd_hist` | Standard MACD triple. |
| `macdfix` | `macdfix`, `macdfix_signal`, `macdfix_hist` | Fixed-fast/slow MACD variant. |
| `macdext` | `macdext`, `macdext_signal`, `macdext_hist` | MACD with explicit MA-type controls. |
| `mom` | `mom_10` | Momentum over `period`. |
| `roc` | `roc_10` | Rate of change in percent-style form. |
| `rocp` | `rocp_10` | Rate of change percentage as decimal fraction. |
| `rocr` | `rocr_10` | Rate of change ratio. |
| `rocr100` | `rocr100_10` | Ratio scaled by 100. |
| `rsi` | `rsi_14` | TA-Lib RSI wrapper. |
| `rsi_sma` | `rsi_sma_14` | SMA-based RSI-style helper. |
| `wilder_rsi` | `wilder_rsi_14` | Wilder-smoothed RSI variant. |
| `cmo` | `cmo_14` | Chande Momentum Oscillator. |
| `ultosc` | `ultosc_7_14_28` | Ultimate Oscillator over three windows. |
| `willr` | `willr_14` | Williams %R. |
| `stoch` | `stoch_slowk`, `stoch_slowd` | TA-Lib stochastic oscillator. |
| `stochf` | `stochf_fastk`, `stochf_fastd` | Fast stochastic variant. |
| `stochrsi` | `stochrsi_fastk`, `stochrsi_fastd` | Stochastic RSI. |
| `stochastic_oscillator` | `stoch_k`, `stoch_d` | Native rolling-high/low implementation. |
| `trix` | `trix_30` | Triple-smoothed momentum oscillator. |
| `tsf` | `tsf_14` | Time series forecast. |
| `linearreg` | `linearreg_14` | Rolling linear regression level. |
| `linearreg_angle` | `linearreg_angle_14` | Regression angle. |
| `linearreg_intercept` | `linearreg_intercept_14` | Regression intercept. |
| `linearreg_slope` | `linearreg_slope_14` | Regression slope. |
| `ht_dcperiod` | `ht_dcperiod` | Dominant cycle period. |
| `ht_dcphase` | `ht_dcphase` | Dominant cycle phase. |
| `ht_phasor` | `ht_phasor_inphase`, `ht_phasor_quadrature` | Hilbert phasor pair. |
| `ht_sine` | `ht_sine`, `ht_sine_lead` | Hilbert sine pair. |
| `ht_trendmode` | `ht_trendmode` | Trend-versus-cycle mode flag. |

## Volatility, Bands, And Range

These helpers quantify range, dispersion, band structure, or stop placement.

| Function | Adds by default | Notes |
|---|---|---|
| `atr` | `atr_14` | Average True Range. |
| `natr` | `natr_14` | Normalized ATR. |
| `trange` | `trange` | True range. |
| `stddev` | `stddev_5_1` | Rolling standard deviation. |
| `var` | `var_5_1` | Rolling variance. |
| `bbands` | `bbands_upper`, `bbands_middle`, `bbands_lower` | TA-Lib-style Bollinger Bands with configurable `ma_type`. |
| `bollinger_bands` | `bb_middle`, `bb_upper`, `bb_lower` | Native SMA-based Bollinger Bands helper. |
| `bollinger_position` | `bollinger_position` | Requires `close`, `bb_upper`, and `bb_lower` to already exist. |
| `rolling_volatility` | `close_volatility_12` | Rolling volatility over a single column. |
| `sma_deviation_std` | `sma30_dev_std` | Deviation from SMA normalized by rolling dispersion. |
| `sar` | `sar` | Parabolic SAR. |
| `sarext` | `sarext` | Extended SAR with separate long and short parameters. |

## Volume, Breadth, And Bar Diagnostics

These helpers focus on participation, bar structure, or direct return transforms.

| Function | Adds by default | Notes |
|---|---|---|
| `ad` | `ad` | Chaikin Accumulation/Distribution line. |
| `adosc` | `adosc_3_10` | Chaikin A/D oscillator. |
| `obv` | `obv` | On-Balance Volume. |
| `mfi` | `mfi_14` | Money Flow Index. |
| `bop` | `bop` | Balance of Power. |
| `price_change_pct` | `price_change_pct_1` | Point-to-point percent change. |
| `returns` | `returns` | Single-step simple returns helper. |
| `body_pct` | `body_pct` | Candle body size as a share of the full range. |
| `window_return` | `ret_24` | Return over a wider rolling window. |

## Candlestick Patterns

All candlestick pattern helpers:

- expect OHLC columns
- append one integer column named after the function
- are best treated as event flags rather than smooth continuous features

Only three patterns expose an extra parameter today.

| Function | Adds by default | Extra params |
|---|---|---|
| `cdl2crows` | `cdl2crows` | None |
| `cdl3blackcrows` | `cdl3blackcrows` | None |
| `cdl3inside` | `cdl3inside` | None |
| `cdl3linestrike` | `cdl3linestrike` | None |
| `cdl3starsinsouth` | `cdl3starsinsouth` | None |
| `cdl3whitesoldiers` | `cdl3whitesoldiers` | None |
| `cdlabandonedbaby` | `cdlabandonedbaby` | `penetration=0.3` |
| `cdladvancedblock` | `cdladvancedblock` | None |
| `cdlbelthold` | `cdlbelthold` | None |
| `cdlclosingmarubozu` | `cdlclosingmarubozu` | None |
| `cdlconcealbabyswall` | `cdlconcealbabyswall` | None |
| `cdlcounterattack` | `cdlcounterattack` | None |
| `cdldarkcloudcover` | `cdldarkcloudcover` | `penetration=0.5` |
| `cdldoji` | `cdldoji` | None |
| `cdldragonflydoji` | `cdldragonflydoji` | None |
| `cdlengulfing` | `cdlengulfing` | None |
| `cdlgravestonedoji` | `cdlgravestonedoji` | None |
| `cdlhammer` | `cdlhammer` | None |
| `cdlhangingman` | `cdlhangingman` | None |
| `cdlharami` | `cdlharami` | None |
| `cdlharamicross` | `cdlharamicross` | None |
| `cdlhighwave` | `cdlhighwave` | None |
| `cdlhikkake` | `cdlhikkake` | None |
| `cdlhikkakemod` | `cdlhikkakemod` | None |
| `cdlhomingpigeon` | `cdlhomingpigeon` | None |
| `cdlidentical3crows` | `cdlidentical3crows` | None |
| `cdlinvertedhammer` | `cdlinvertedhammer` | None |
| `cdlladderbottom` | `cdlladderbottom` | None |
| `cdllongleggeddoji` | `cdllongleggeddoji` | None |
| `cdllongline` | `cdllongline` | None |
| `cdlmarubozu` | `cdlmarubozu` | None |
| `cdlmatchinglow` | `cdlmatchinglow` | None |
| `cdlmathold` | `cdlmathold` | `penetration=0.5` |
| `cdlonneck` | `cdlonneck` | None |
| `cdlpiercing` | `cdlpiercing` | None |
| `cdlrickshawman` | `cdlrickshawman` | None |
| `cdlrisefall3methods` | `cdlrisefall3methods` | None |
| `cdlseparatinglines` | `cdlseparatinglines` | None |
| `cdlshootingstar` | `cdlshootingstar` | None |
| `cdlshortline` | `cdlshortline` | None |
| `cdlspinningtop` | `cdlspinningtop` | None |
| `cdlstalledpattern` | `cdlstalledpattern` | None |
| `cdlsticksandwich` | `cdlsticksandwich` | None |
| `cdltakuri` | `cdltakuri` | None |
| `cdlthrusting` | `cdlthrusting` | None |
| `cdltristar` | `cdltristar` | None |
| `cdlunique3river` | `cdlunique3river` | None |

## Choosing Between Similar Helpers

- Use `bbands` when you want TA-Lib parity and configurable moving-average type.
- Use `bollinger_bands` when you want a lightweight native SMA-based band helper.
- Use `stoch` or `stochf` when you want TA-Lib parity. Use `stochastic_oscillator` when you want a simpler native rolling implementation.
- Use `sma` when you want the canonical moving average helper used across Limen manifests. Use `ma` when `ma_type` itself is part of the search space.
- Use `returns`, `price_change_pct`, and `window_return` differently: one-bar simple returns, configurable percent change, and wider-window return summaries respectively.

## Read Next

- [Features](Features.md) for higher-level derived signals built on top of raw bars or indicators
- [Experiment Manifest](Experiment-Manifest.md) for how to wire indicators into a manifest pipeline
- [Single File Decoder](Single-File-Decoder.md) for where indicator selection fits inside an SFD
