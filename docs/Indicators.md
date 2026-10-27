# Indicators

The [configuration constraints](Features.md#configuration-constraints) apply to direct indicator calls as well as feature helpers.

Indicators are Limen's lowest-level signal helpers. They take a kline-style `pl.DataFrame`, append one or more derived columns, and return the resulting frame for ordinary Python preparation or the declared Compiled pipeline.

This page covers indicator selection, naming conventions, and default added columns. For higher-level derived signals, see [Features](Features.md).

## Conventions

- Indicators append columns. They do not replace the input frame with a reduced schema.
- Helpers assume the standard OHLCV names: `open`, `high`, `low`, `close`, `volume`.
- TA-Lib-backed wrappers treat TA-Lib parity as the source of truth. If Limen and TA-Lib diverge, TA-Lib wins unless there is a deliberate, documented reason not to.
- Output column names encode key defaults, such as `rsi_14` or `apo_12_26_0`.
- Candlestick pattern helpers add a same-named integer column following TA-Lib's positive bullish, negative bearish, zero neutral convention.
- Legacy compatibility helpers intentionally expose legacy behavior. `sma` adds both `sma_{period}` and `{price_col}_sma_{period}`.

## Use an indicator in the projects

Python Research can compute a column directly or call a Limen indicator. The [Python project](examples/python-research/README.md) owns this decimal momentum expression and uses it in fitting, validation and inference:

<!-- example: indicators-python-momentum source=docs/examples/python-research/research.py region=momentum_feature scenario=selection interface=python role=executed -->
```python
def features(bars, window):
    return bars.with_columns(
        (pl.col("close") / pl.col("close").shift(window) - 1).alias("momentum")
    )
```

The [Compiled project](examples/compiled-research/README.md) declares a one-period rate of change in its recorded pipeline. The indicator returns percent units; it is not the Python project's decimal return column.

<!-- example: indicators-compiled-momentum source=docs/examples/compiled-research/selection.yaml region=momentum_indicator scenario=selection interface=compiled role=executed -->
```yaml
    indicators:
      - func: limen.indicators.roc
        params:
          period: 1
          group: momentum
```

Both excerpts belong to the complete selection programs. Their source footers lead to the inputs, current wheel execution and semantic assertions. Python can use shared indicators in caller-owned preparation; Compiled Research declares them in YAML and executes them through the CLI.

### Rate-of-change signature

The function below is a current package signature. It describes an interface; it is not a standalone research program.

<!-- reference: indicators-roc-signature source=limen/indicators/roc.py region=signature:roc role=signature -->
```python
def roc(data: pl.DataFrame, price_col: str='close', period: int=10) -> pl.DataFrame:
    ...
```

## Family index

- `Price and moving averages`: direct price transforms and smoothing helpers.
- `Momentum and oscillators`: rate-of-change, oscillator, regression, and Hilbert-style cycle helpers.
- `Volatility, bands, and range`: dispersion, band, and stop-style helpers.
- `Volume, breadth, and bar diagnostics`: flow, bar-body, and return-style helpers.
- `Candlestick patterns`: TA-Lib-style pattern flags over OHLC bars.

## Price and moving averages

These helpers take either a single `price_col` or basic OHLC inputs and add smoothed or transformed price views.

| Function | Adds by default | Notes |
|---|---|---|
| `avgprice` | `avgprice` | Average OHLC price. |
| `medprice` | `medprice` | Median of high and low. |
| `midprice` | `midprice_14` | Rolling midpoint of high and low over `period`. |
| `typprice` | `typprice` | Mean of high, low, and close. |
| `wclprice` | `wclprice` | Weighted close price. |
| `midpoint` | `midpoint_14` | Rolling midpoint of a single `price_col`. |
| `sma` | `sma_30`, `close_sma_30` | Adds both the canonical and compatibility alias columns. |
| `ema` | `ema_30` | Exponential moving average. |
| `wma` | `wma_30` | Weighted moving average. |
| `dema` | `dema_30` | Double EMA. |
| `tema` | `tema_30` | Triple EMA. |
| `trima` | `trima_30` | Triangular moving average. |
| `t3` | `t3_5_0.7` | T3 moving average with `period` and `vfactor`. |
| `kama` | `kama_30` | Kaufman adaptive moving average. |
| `mama` | `mama`, `fama` | MESA adaptive moving average pair. |
| `ma` | `ma_30_0` | Generic moving average with explicit `ma_type`. |
| `ht_trendline` | `ht_trendline` | Hilbert Transform instantaneous trendline. |

## Momentum and oscillators

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
| `cci` | `cci_14` | Commodity Channel Index. |
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

## Volatility, bands, and range

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

## Volume, breadth, and bar diagnostics

These helpers focus on participation, bar structure, or direct return transforms.

| Function | Adds by default | Notes |
|---|---|---|
| `ad` | `ad` | Chaikin Accumulation/Distribution line. |
| `adosc` | `adosc_3_10` | Chaikin A/D oscillator. |
| `obv` | `obv` | On-Balance Volume. |
| `mfi` | `mfi_14` | Money Flow Index. |
| `bop` | `bop` | Balance of Power. |
| `price_change_pct` | `price_change_pct_1` | Point-to-point percent change. |
| `returns` | `returns` | Single-step returns helper. |
| `body_pct` | `body_pct` | Candle body size as a share of the full range. |
| `window_return` | `ret_24` | Return over a wider rolling window. |

## Candlestick patterns

All candlestick pattern helpers:

- expect OHLC columns
- append one integer column named after the function
- should be treated as event flags rather than smooth continuous features

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

## Choosing between similar helpers

- Use `bbands` for TA-Lib parity and configurable moving-average type.
- Use `bollinger_bands` for the native SMA-based band helper.
- Use `stoch` or `stochf` for TA-Lib parity. Use `stochastic_oscillator` for the native rolling implementation.
- Use `sma` for the canonical moving average helper used across Limen manifests. Use `ma` when `ma_type` itself is part of the search space.
- Use `returns`, `price_change_pct`, and `window_return` for one-bar returns, configurable percent change, and wider-window return summaries respectively.

## Read next

- [Features](Features.md) for higher-level derived signals built on top of raw bars or indicators
- [Experiment Manifest](Experiment-Manifest.md) for how to wire indicators into a manifest pipeline
- [Single-File Decoder](Single-File-Decoder.md) for where indicator selection fits inside an SFD

## Calculation cost

The [indicator performance appendix](Developer/Indicator-Performance.md) retains the measured calculation costs, environments, numerical-parity boundaries and reproduction commands. Those measurements describe computation on their recorded inputs, not the returns of a research strategy.

## Callable contracts

These entries give the exact public signatures and the source-owned argument, return and failure contracts. Use the project examples above for the research context; a signature is reference evidence, not a claim that every option ran in the base experiment.

### `limen.indicators.ad`

<!-- reference: api-limen-indicators-ad source=limen/indicators/ad.py region=signature:ad role=signature -->
```python
def ad(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute Chaikin Accumulation/Distribution (A/D) line.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high/low/close/volume columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| volume_col (`str`) | Column name for volume |

**Returns:** pl.DataFrame: The input data with a new column 'ad'

### `limen.indicators.adosc`

<!-- reference: api-limen-indicators-adosc source=limen/indicators/adosc.py region=signature:adosc role=signature -->
```python
def adosc(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', volume_col: str='volume', fast_period: int=3, slow_period: int=10) -> pl.DataFrame:
    ...
```

Compute Chaikin A/D Oscillator (ADOSC).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high/low/close/volume columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| volume_col (`str`) | Column name for volume |
| fast_period (`int`) | Fast EMA period |
| slow_period (`int`) | Slow EMA period |

**Returns:** pl.DataFrame: The input data with a new column 'adosc_{fast_period}_{slow_period}'

### `limen.indicators.apo`

<!-- reference: api-limen-indicators-apo source=limen/indicators/apo.py region=signature:apo role=signature -->
```python
def apo(data: pl.DataFrame, price_col: str='close', fast_period: int=12, slow_period: int=26, ma_type: int=0) -> pl.DataFrame:
    ...
```

Compute Absolute Price Oscillator (APO).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| fast_period (`int`) | Number of periods for fast MA (2..100000) |
| slow_period (`int`) | Number of periods for slow MA (2..100000, must be greater than fast_period) |
| ma_type (`int`) | TA-Lib MA type (0..8) |

**Returns:** pl.DataFrame: The input data with a new column 'apo_{fast_period}_{slow_period}_{ma_type}'

### `limen.indicators.atr`

<!-- reference: api-limen-indicators-atr source=limen/indicators/atr.py region=signature:atr role=signature -->
```python
def atr(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Average True Range (ATR).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high/low/close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| period (`int`) | Number of periods for ATR calculation |

**Returns:** pl.DataFrame: The input data with a new column 'atr_{period}'

### `limen.indicators.avgprice`

<!-- reference: api-limen-indicators-avgprice source=limen/indicators/avgprice.py region=signature:avgprice role=signature -->
```python
def avgprice(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute Average Price.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with OHLC columns |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'avgprice'

### `limen.indicators.bbands`

<!-- reference: api-limen-indicators-bbands source=limen/indicators/bbands.py region=signature:bbands role=signature -->
```python
def bbands(data: pl.DataFrame, price_col: str='close', period: int=5, nb_dev_up: float=2.0, nb_dev_dn: float=2.0, ma_type: int=0) -> pl.DataFrame:
    ...
```

Compute Bollinger Bands (upper/middle/lower).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Rolling window length |
| nb_dev_up (`float`) | Upper-band deviation multiplier |
| nb_dev_dn (`float`) | Lower-band deviation multiplier |
| ma_type (`int`) | TA-Lib MA type |

**Returns:** pl.DataFrame: Input data with 'bbands_upper', 'bbands_middle', 'bbands_lower'

### `limen.indicators.body_pct`

<!-- reference: api-limen-indicators-body-pct source=limen/indicators/body_pct.py region=signature:body_pct role=signature -->
```python
def body_pct(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute the body percentage (candle body size relative to open).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'open' and 'close' columns |

**Returns:** pl.DataFrame: The input data with a new column 'body_pct'

### `limen.indicators.bollinger_bands`

<!-- reference: api-limen-indicators-bollinger-bands source=limen/indicators/bollinger_bands.py region=signature:bollinger_bands role=signature -->
```python
def bollinger_bands(df: pl.DataFrame, price_col: str='close', window: int=20, num_std: float=2.0) -> pl.DataFrame:
    ...
```

Compute Bollinger Bands using Simple Moving Average (SMA).

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'close' column |
| price_col (`str`) | Column name used for Bollinger Band calculation |
| window (`int`) | Number of periods for SMA and standard deviation calculation |
| num_std (`float`) | Number of standard deviations for upper and lower bands |

**Returns:** pl.DataFrame: The input data with three new columns: 'bb_middle', 'bb_upper', and 'bb_lower'

### `limen.indicators.bollinger_position`

<!-- reference: api-limen-indicators-bollinger-position source=limen/indicators/bollinger_position.py region=signature:bollinger_position role=signature -->
```python
def bollinger_position(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute price position within Bollinger Bands as percentage.

Returns 0 when price is at lower band, 1 when at upper band, 0.5 at middle.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close', 'bb_upper', 'bb_lower' columns |

**Returns:** pl.DataFrame: The input data with a new column 'bollinger_position'

### `limen.indicators.bop`

<!-- reference: api-limen-indicators-bop source=limen/indicators/bop.py region=signature:bop role=signature -->
```python
def bop(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute Balance of Power (BOP).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with OHLC columns |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'bop'

### `limen.indicators.cci`

<!-- reference: api-limen-indicators-cci source=limen/indicators/cci.py region=signature:cci role=signature -->
```python
def cci(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Append CCI with the original chronological mean/deviation sums (period 2..100000).

### `limen.indicators.cdl2crows`

<!-- reference: api-limen-indicators-cdl2crows source=limen/indicators/cdl2crows.py region=signature:cdl2crows role=signature -->
```python
def cdl2crows(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdl3blackcrows`

<!-- reference: api-limen-indicators-cdl3blackcrows source=limen/indicators/cdl3blackcrows.py region=signature:cdl3blackcrows role=signature -->
```python
def cdl3blackcrows(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdl3inside`

<!-- reference: api-limen-indicators-cdl3inside source=limen/indicators/cdl3inside.py region=signature:cdl3inside role=signature -->
```python
def cdl3inside(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdl3linestrike`

<!-- reference: api-limen-indicators-cdl3linestrike source=limen/indicators/cdl3linestrike.py region=signature:cdl3linestrike role=signature -->
```python
def cdl3linestrike(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdl3starsinsouth`

<!-- reference: api-limen-indicators-cdl3starsinsouth source=limen/indicators/cdl3starsinsouth.py region=signature:cdl3starsinsouth role=signature -->
```python
def cdl3starsinsouth(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdl3whitesoldiers`

<!-- reference: api-limen-indicators-cdl3whitesoldiers source=limen/indicators/cdl3whitesoldiers.py region=signature:cdl3whitesoldiers role=signature -->
```python
def cdl3whitesoldiers(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlabandonedbaby`

<!-- reference: api-limen-indicators-cdlabandonedbaby source=limen/indicators/cdlabandonedbaby.py region=signature:cdlabandonedbaby role=signature -->
```python
def cdlabandonedbaby(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close', penetration: float=0.3) -> pl.DataFrame:
    ...
```

### `limen.indicators.cdladvancedblock`

<!-- reference: api-limen-indicators-cdladvancedblock source=limen/indicators/cdladvancedblock.py region=signature:cdladvancedblock role=signature -->
```python
def cdladvancedblock(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlbelthold`

<!-- reference: api-limen-indicators-cdlbelthold source=limen/indicators/cdlbelthold.py region=signature:cdlbelthold role=signature -->
```python
def cdlbelthold(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlclosingmarubozu`

<!-- reference: api-limen-indicators-cdlclosingmarubozu source=limen/indicators/cdlclosingmarubozu.py region=signature:cdlclosingmarubozu role=signature -->
```python
def cdlclosingmarubozu(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlconcealbabyswall`

<!-- reference: api-limen-indicators-cdlconcealbabyswall source=limen/indicators/cdlconcealbabyswall.py region=signature:cdlconcealbabyswall role=signature -->
```python
def cdlconcealbabyswall(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlcounterattack`

<!-- reference: api-limen-indicators-cdlcounterattack source=limen/indicators/cdlcounterattack.py region=signature:cdlcounterattack role=signature -->
```python
def cdlcounterattack(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdldarkcloudcover`

<!-- reference: api-limen-indicators-cdldarkcloudcover source=limen/indicators/cdldarkcloudcover.py region=signature:cdldarkcloudcover role=signature -->
```python
def cdldarkcloudcover(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close', penetration: float=0.5) -> pl.DataFrame:
    ...
```

### `limen.indicators.cdldoji`

<!-- reference: api-limen-indicators-cdldoji source=limen/indicators/cdldoji.py region=signature:cdldoji role=signature -->
```python
def cdldoji(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdldragonflydoji`

<!-- reference: api-limen-indicators-cdldragonflydoji source=limen/indicators/cdldragonflydoji.py region=signature:cdldragonflydoji role=signature -->
```python
def cdldragonflydoji(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlengulfing`

<!-- reference: api-limen-indicators-cdlengulfing source=limen/indicators/cdlengulfing.py region=signature:cdlengulfing role=signature -->
```python
def cdlengulfing(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlgravestonedoji`

<!-- reference: api-limen-indicators-cdlgravestonedoji source=limen/indicators/cdlgravestonedoji.py region=signature:cdlgravestonedoji role=signature -->
```python
def cdlgravestonedoji(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlhammer`

<!-- reference: api-limen-indicators-cdlhammer source=limen/indicators/cdlhammer.py region=signature:cdlhammer role=signature -->
```python
def cdlhammer(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlhangingman`

<!-- reference: api-limen-indicators-cdlhangingman source=limen/indicators/cdlhangingman.py region=signature:cdlhangingman role=signature -->
```python
def cdlhangingman(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlharami`

<!-- reference: api-limen-indicators-cdlharami source=limen/indicators/cdlharami.py region=signature:cdlharami role=signature -->
```python
def cdlharami(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlharamicross`

<!-- reference: api-limen-indicators-cdlharamicross source=limen/indicators/cdlharamicross.py region=signature:cdlharamicross role=signature -->
```python
def cdlharamicross(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlhighwave`

<!-- reference: api-limen-indicators-cdlhighwave source=limen/indicators/cdlhighwave.py region=signature:cdlhighwave role=signature -->
```python
def cdlhighwave(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlhikkake`

<!-- reference: api-limen-indicators-cdlhikkake source=limen/indicators/cdlhikkake.py region=signature:cdlhikkake role=signature -->
```python
def cdlhikkake(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlhikkakemod`

<!-- reference: api-limen-indicators-cdlhikkakemod source=limen/indicators/cdlhikkakemod.py region=signature:cdlhikkakemod role=signature -->
```python
def cdlhikkakemod(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlhomingpigeon`

<!-- reference: api-limen-indicators-cdlhomingpigeon source=limen/indicators/cdlhomingpigeon.py region=signature:cdlhomingpigeon role=signature -->
```python
def cdlhomingpigeon(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlidentical3crows`

<!-- reference: api-limen-indicators-cdlidentical3crows source=limen/indicators/cdlidentical3crows.py region=signature:cdlidentical3crows role=signature -->
```python
def cdlidentical3crows(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlinvertedhammer`

<!-- reference: api-limen-indicators-cdlinvertedhammer source=limen/indicators/cdlinvertedhammer.py region=signature:cdlinvertedhammer role=signature -->
```python
def cdlinvertedhammer(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlladderbottom`

<!-- reference: api-limen-indicators-cdlladderbottom source=limen/indicators/cdlladderbottom.py region=signature:cdlladderbottom role=signature -->
```python
def cdlladderbottom(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdllongleggeddoji`

<!-- reference: api-limen-indicators-cdllongleggeddoji source=limen/indicators/cdllongleggeddoji.py region=signature:cdllongleggeddoji role=signature -->
```python
def cdllongleggeddoji(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdllongline`

<!-- reference: api-limen-indicators-cdllongline source=limen/indicators/cdllongline.py region=signature:cdllongline role=signature -->
```python
def cdllongline(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlmarubozu`

<!-- reference: api-limen-indicators-cdlmarubozu source=limen/indicators/cdlmarubozu.py region=signature:cdlmarubozu role=signature -->
```python
def cdlmarubozu(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlmatchinglow`

<!-- reference: api-limen-indicators-cdlmatchinglow source=limen/indicators/cdlmatchinglow.py region=signature:cdlmatchinglow role=signature -->
```python
def cdlmatchinglow(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlmathold`

<!-- reference: api-limen-indicators-cdlmathold source=limen/indicators/cdlmathold.py region=signature:cdlmathold role=signature -->
```python
def cdlmathold(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close', penetration: float=0.5) -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlonneck`

<!-- reference: api-limen-indicators-cdlonneck source=limen/indicators/cdlonneck.py region=signature:cdlonneck role=signature -->
```python
def cdlonneck(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlpiercing`

<!-- reference: api-limen-indicators-cdlpiercing source=limen/indicators/cdlpiercing.py region=signature:cdlpiercing role=signature -->
```python
def cdlpiercing(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlrickshawman`

<!-- reference: api-limen-indicators-cdlrickshawman source=limen/indicators/cdlrickshawman.py region=signature:cdlrickshawman role=signature -->
```python
def cdlrickshawman(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlrisefall3methods`

<!-- reference: api-limen-indicators-cdlrisefall3methods source=limen/indicators/cdlrisefall3methods.py region=signature:cdlrisefall3methods role=signature -->
```python
def cdlrisefall3methods(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlseparatinglines`

<!-- reference: api-limen-indicators-cdlseparatinglines source=limen/indicators/cdlseparatinglines.py region=signature:cdlseparatinglines role=signature -->
```python
def cdlseparatinglines(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlshootingstar`

<!-- reference: api-limen-indicators-cdlshootingstar source=limen/indicators/cdlshootingstar.py region=signature:cdlshootingstar role=signature -->
```python
def cdlshootingstar(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlshortline`

<!-- reference: api-limen-indicators-cdlshortline source=limen/indicators/cdlshortline.py region=signature:cdlshortline role=signature -->
```python
def cdlshortline(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlspinningtop`

<!-- reference: api-limen-indicators-cdlspinningtop source=limen/indicators/cdlspinningtop.py region=signature:cdlspinningtop role=signature -->
```python
def cdlspinningtop(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlstalledpattern`

<!-- reference: api-limen-indicators-cdlstalledpattern source=limen/indicators/cdlstalledpattern.py region=signature:cdlstalledpattern role=signature -->
```python
def cdlstalledpattern(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlsticksandwich`

<!-- reference: api-limen-indicators-cdlsticksandwich source=limen/indicators/cdlsticksandwich.py region=signature:cdlsticksandwich role=signature -->
```python
def cdlsticksandwich(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdltakuri`

<!-- reference: api-limen-indicators-cdltakuri source=limen/indicators/cdltakuri.py region=signature:cdltakuri role=signature -->
```python
def cdltakuri(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlthrusting`

<!-- reference: api-limen-indicators-cdlthrusting source=limen/indicators/cdlthrusting.py region=signature:cdlthrusting role=signature -->
```python
def cdlthrusting(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdltristar`

<!-- reference: api-limen-indicators-cdltristar source=limen/indicators/cdltristar.py region=signature:cdltristar role=signature -->
```python
def cdltristar(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cdlunique3river`

<!-- reference: api-limen-indicators-cdlunique3river source=limen/indicators/cdlunique3river.py region=signature:cdlunique3river role=signature -->
```python
def cdlunique3river(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

### `limen.indicators.cmo`

<!-- reference: api-limen-indicators-cmo source=limen/indicators/cmo.py region=signature:cmo role=signature -->
```python
def cmo(data: pl.DataFrame, price_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Chande Momentum Oscillator (CMO).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (2..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'cmo_{period}'

### `limen.indicators.dema`

<!-- reference: api-limen-indicators-dema source=limen/indicators/dema.py region=signature:dema role=signature -->
```python
def dema(data: pl.DataFrame, price_col: str='close', period: int=30) -> pl.DataFrame:
    ...
```

Compute Double Exponential Moving Average (DEMA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'dema_{period}'

### `limen.indicators.ema`

<!-- reference: api-limen-indicators-ema source=limen/indicators/ema.py region=signature:ema role=signature -->
```python
def ema(data: pl.DataFrame, price_col: str='close', period: int=30) -> pl.DataFrame:
    ...
```

Compute Exponential Moving Average (EMA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'ema_{period}'

### `limen.indicators.ht_dcperiod`

<!-- reference: api-limen-indicators-ht-dcperiod source=limen/indicators/ht_dcperiod.py region=signature:ht_dcperiod role=signature -->
```python
def ht_dcperiod(data: pl.DataFrame, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute Hilbert Transform - Dominant Cycle Period.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |

**Returns:** pl.DataFrame: The input data with a new column 'ht_dcperiod'

### `limen.indicators.ht_dcphase`

<!-- reference: api-limen-indicators-ht-dcphase source=limen/indicators/ht_dcphase.py region=signature:ht_dcphase role=signature -->
```python
def ht_dcphase(data: pl.DataFrame, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute Hilbert Transform - Dominant Cycle Phase.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |

**Returns:** pl.DataFrame: The input data with a new column 'ht_dcphase'

### `limen.indicators.ht_phasor`

<!-- reference: api-limen-indicators-ht-phasor source=limen/indicators/ht_phasor.py region=signature:ht_phasor role=signature -->
```python
def ht_phasor(data: pl.DataFrame, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute Hilbert Transform - Phasor Components.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |

**Returns:** pl.DataFrame: The input data with new columns 'ht_phasor_inphase' and

'ht_phasor_quadrature'

### `limen.indicators.ht_sine`

<!-- reference: api-limen-indicators-ht-sine source=limen/indicators/ht_sine.py region=signature:ht_sine role=signature -->
```python
def ht_sine(data: pl.DataFrame, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute Hilbert Transform - SineWave.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |

**Returns:** pl.DataFrame: The input data with new columns 'ht_sine' and

'ht_sine_lead'

### `limen.indicators.ht_trendline`

<!-- reference: api-limen-indicators-ht-trendline source=limen/indicators/ht_trendline.py region=signature:ht_trendline role=signature -->
```python
def ht_trendline(data: pl.DataFrame, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute Hilbert Transform - Instantaneous Trendline.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |

**Returns:** pl.DataFrame: The input data with a new column 'ht_trendline'

### `limen.indicators.ht_trendmode`

<!-- reference: api-limen-indicators-ht-trendmode source=limen/indicators/ht_trendmode.py region=signature:ht_trendmode role=signature -->
```python
def ht_trendmode(data: pl.DataFrame, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute Hilbert Transform - Trend vs Cycle Mode.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |

**Returns:** pl.DataFrame: The input data with a new column 'ht_trendmode'

### `limen.indicators.kama`

<!-- reference: api-limen-indicators-kama source=limen/indicators/kama.py region=signature:kama role=signature -->
```python
def kama(data: pl.DataFrame, price_col: str='close', period: int=30) -> pl.DataFrame:
    ...
```

Compute Kaufman Adaptive Moving Average (KAMA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'kama_{period}'

### `limen.indicators.linearreg`

<!-- reference: api-limen-indicators-linearreg source=limen/indicators/linearreg.py region=signature:linearreg role=signature -->
```python
def linearreg(data: pl.DataFrame, price_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Append the trailing least-squares line value with the original warmup.

### `limen.indicators.linearreg_angle`

<!-- reference: api-limen-indicators-linearreg-angle source=limen/indicators/linearreg_angle.py region=signature:linearreg_angle role=signature -->
```python
def linearreg_angle(data: pl.DataFrame, price_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Linear Regression Angle.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'linearreg_angle_{period}'

### `limen.indicators.linearreg_intercept`

<!-- reference: api-limen-indicators-linearreg-intercept source=limen/indicators/linearreg_intercept.py region=signature:linearreg_intercept role=signature -->
```python
def linearreg_intercept(data: pl.DataFrame, price_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Linear Regression Intercept.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'linearreg_intercept_{period}'

### `limen.indicators.linearreg_slope`

<!-- reference: api-limen-indicators-linearreg-slope source=limen/indicators/linearreg_slope.py region=signature:linearreg_slope role=signature -->
```python
def linearreg_slope(data: pl.DataFrame, price_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Linear Regression Slope.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'linearreg_slope_{period}'

### `limen.indicators.ma`

<!-- reference: api-limen-indicators-ma source=limen/indicators/ma.py region=signature:ma role=signature -->
```python
def ma(data: pl.DataFrame, price_col: str='close', period: int=30, ma_type: int=0) -> pl.DataFrame:
    ...
```

Compute Moving Average with selectable MA type (TA-Lib MA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |
| ma_type (`int`) | TA-Lib MA type |

**Returns:** pl.DataFrame: The input data with a new column 'ma_{period}_{ma_type}'

### `limen.indicators.macd`

<!-- reference: api-limen-indicators-macd source=limen/indicators/macd.py region=signature:macd role=signature -->
```python
def macd(data: pl.DataFrame, price_col: str='close', fast_period: int=12, slow_period: int=26, signal_period: int=9) -> pl.DataFrame:
    ...
```

Compute Moving Average Convergence/Divergence (MACD).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| fast_period (`int`) | Number of periods for fast EMA (2..100000) |
| slow_period (`int`) | Number of periods for slow EMA (2..100000) |
| signal_period (`int`) | Number of periods for signal EMA (1..100000) |

**Returns:** pl.DataFrame: The input data with columns 'macd', 'macd_signal', 'macd_hist'

### `limen.indicators.macdext`

<!-- reference: api-limen-indicators-macdext source=limen/indicators/macdext.py region=signature:macdext role=signature -->
```python
def macdext(data: pl.DataFrame, price_col: str='close', fast_period: int=12, fast_ma_type: int=0, slow_period: int=26, slow_ma_type: int=0, signal_period: int=9, signal_ma_type: int=0) -> pl.DataFrame:
    ...
```

Compute MACD with controllable MA types (MACDEXT).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| fast_period (`int`) | Number of periods for fast MA (2..100000) |
| fast_ma_type (`int`) | TA-Lib MA type for fast MA (0..8) |
| slow_period (`int`) | Number of periods for slow MA (2..100000) |
| slow_ma_type (`int`) | TA-Lib MA type for slow MA (0..8) |
| signal_period (`int`) | Number of periods for signal MA (1..100000) |
| signal_ma_type (`int`) | TA-Lib MA type for signal MA (0..8) |

**Returns:** pl.DataFrame: The input data with columns 'macdext', 'macdext_signal', 'macdext_hist'

### `limen.indicators.macdfix`

<!-- reference: api-limen-indicators-macdfix source=limen/indicators/macdfix.py region=signature:macdfix role=signature -->
```python
def macdfix(data: pl.DataFrame, price_col: str='close', signal_period: int=9) -> pl.DataFrame:
    ...
```

Compute MACD Fix 12/26 (MACDFIX).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| signal_period (`int`) | Number of periods for signal EMA (1..100000) |

**Returns:** pl.DataFrame: The input data with columns 'macdfix', 'macdfix_signal', 'macdfix_hist'

### `limen.indicators.mama`

<!-- reference: api-limen-indicators-mama source=limen/indicators/mama.py region=signature:mama role=signature -->
```python
def mama(data: pl.DataFrame, price_col: str='close', fast_limit: float=0.5, slow_limit: float=0.05) -> pl.DataFrame:
    ...
```

Compute MESA Adaptive Moving Average (MAMA) and Following Adaptive MA (FAMA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| fast_limit (`float`) | Upper adaptive limit |
| slow_limit (`float`) | Lower adaptive limit |

**Returns:** pl.DataFrame: The input data with new columns 'mama' and 'fama'

### `limen.indicators.medprice`

<!-- reference: api-limen-indicators-medprice source=limen/indicators/medprice.py region=signature:medprice role=signature -->
```python
def medprice(data: pl.DataFrame, high_col: str='high', low_col: str='low') -> pl.DataFrame:
    ...
```

Compute Median Price.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high and low columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |

**Returns:** pl.DataFrame: The input data with a new column 'medprice'

### `limen.indicators.mfi`

<!-- reference: api-limen-indicators-mfi source=limen/indicators/mfi.py region=signature:mfi role=signature -->
```python
def mfi(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', volume_col: str='volume', period: int=14) -> pl.DataFrame:
    ...
```

Compute Money Flow Index (MFI).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high/low/close/volume columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| volume_col (`str`) | Column name for volume |
| period (`int`) | MFI period |

**Returns:** pl.DataFrame: The input data with a new column 'mfi_{period}'

### `limen.indicators.midpoint`

<!-- reference: api-limen-indicators-midpoint source=limen/indicators/midpoint.py region=signature:midpoint role=signature -->
```python
def midpoint(data: pl.DataFrame, price_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute MidPoint over period.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (2..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'midpoint_{period}'

### `limen.indicators.midprice`

<!-- reference: api-limen-indicators-midprice source=limen/indicators/midprice.py region=signature:midprice role=signature -->
```python
def midprice(data: pl.DataFrame, high_col: str='high', low_col: str='low', period: int=14) -> pl.DataFrame:
    ...
```

Compute the trailing high/low midpoint over period (2..100000).

### `limen.indicators.mom`

<!-- reference: api-limen-indicators-mom source=limen/indicators/mom.py region=signature:mom role=signature -->
```python
def mom(data: pl.DataFrame, price_col: str='close', period: int=10) -> pl.DataFrame:
    ...
```

Compute Momentum (MOM).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (1..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'mom_{period}'

### `limen.indicators.natr`

<!-- reference: api-limen-indicators-natr source=limen/indicators/natr.py region=signature:natr role=signature -->
```python
def natr(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Normalized Average True Range (NATR).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high/low/close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| period (`int`) | Number of periods for NATR calculation |

**Returns:** pl.DataFrame: The input data with a new column 'natr_{period}'

### `limen.indicators.obv`

<!-- reference: api-limen-indicators-obv source=limen/indicators/obv.py region=signature:obv role=signature -->
```python
def obv(data: pl.DataFrame, price_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute On-Balance Volume (OBV).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with price and volume columns |
| price_col (`str`) | Column name for price series |
| volume_col (`str`) | Column name for volume series |

**Returns:** pl.DataFrame: The input data with a new column 'obv'

### `limen.indicators.ppo`

<!-- reference: api-limen-indicators-ppo source=limen/indicators/ppo.py region=signature:ppo role=signature -->
```python
def ppo(data: pl.DataFrame, price_col: str='close', fast_period: int=12, slow_period: int=26, ma_type: int=0) -> pl.DataFrame:
    ...
```

Compute Percentage Price Oscillator (PPO).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| fast_period (`int`) | Number of periods for fast MA (2..100000) |
| slow_period (`int`) | Number of periods for slow MA (2..100000, must be greater than fast_period) |
| ma_type (`int`) | TA-Lib MA type (0..8) |

**Returns:** pl.DataFrame: The input data with a new column 'ppo_{fast_period}_{slow_period}_{ma_type}'

### `limen.indicators.price_change_pct`

<!-- reference: api-limen-indicators-price-change-pct source=limen/indicators/price_change_pct.py region=signature:price_change_pct role=signature -->
```python
def price_change_pct(data: pl.DataFrame, period: int=1) -> pl.DataFrame:
    ...
```

Compute price change percentage over a specific period.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Number of periods to look back |

**Returns:** pl.DataFrame: The input data with a new column 'price_change_pct_{period}'

### `limen.indicators.returns`

<!-- reference: api-limen-indicators-returns source=limen/indicators/returns.py region=signature:returns role=signature -->
```python
def returns(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute period-over-period returns of close prices.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |

**Returns:** pl.DataFrame: The input data with a new column 'returns'

### `limen.indicators.roc`

<!-- reference: api-limen-indicators-roc source=limen/indicators/roc.py region=signature:roc role=signature -->
```python
def roc(data: pl.DataFrame, price_col: str='close', period: int=10) -> pl.DataFrame:
    ...
```

Compute Rate of Change (ROC): ((price / prev_price) - 1) * 100.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (1..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'roc_{period}'

### `limen.indicators.rocp`

<!-- reference: api-limen-indicators-rocp source=limen/indicators/rocp.py region=signature:rocp role=signature -->
```python
def rocp(data: pl.DataFrame, price_col: str='close', period: int=10) -> pl.DataFrame:
    ...
```

Compute Rate of Change Percentage (ROCP): (price - prev_price) / prev_price.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (1..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'rocp_{period}'

### `limen.indicators.rocr`

<!-- reference: api-limen-indicators-rocr source=limen/indicators/rocr.py region=signature:rocr role=signature -->
```python
def rocr(data: pl.DataFrame, price_col: str='close', period: int=10) -> pl.DataFrame:
    ...
```

Compute Rate of Change Ratio (ROCR): price / prev_price.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (1..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'rocr_{period}'

### `limen.indicators.rocr100`

<!-- reference: api-limen-indicators-rocr100 source=limen/indicators/rocr100.py region=signature:rocr100 role=signature -->
```python
def rocr100(data: pl.DataFrame, price_col: str='close', period: int=10) -> pl.DataFrame:
    ...
```

Compute Rate of Change Ratio 100 scale (ROCR100): (price / prev_price) * 100.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (1..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'rocr100_{period}'

### `limen.indicators.rolling_volatility`

<!-- reference: api-limen-indicators-rolling-volatility source=limen/indicators/rolling_volatility.py region=signature:rolling_volatility role=signature -->
```python
def rolling_volatility(data: pl.DataFrame, column: str='close', window: int=12) -> pl.DataFrame:
    ...
```

Compute rolling volatility (standard deviation) over a specified period.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with price/returns column |
| column (`str`) | Column name to calculate volatility on (typically returns) |
| window (`int`) | Number of periods for rolling window calculation |

**Returns:** pl.DataFrame: The input data with a new column '{column}_volatility_{window}'

### `limen.indicators.rsi`

<!-- reference: api-limen-indicators-rsi source=limen/indicators/rsi.py region=signature:rsi role=signature -->
```python
def rsi(data: pl.DataFrame, price_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Relative Strength Index (RSI) using Wilder smoothing.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (2..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'rsi_{period}'

### `limen.indicators.rsi_sma`

<!-- reference: api-limen-indicators-rsi-sma source=limen/indicators/rsi_sma.py region=signature:rsi_sma role=signature -->
```python
def rsi_sma(data: pl.DataFrame, period: int=14) -> pl.DataFrame:
    ...
```

Compute RSI using Simple Moving Average smoothing (not Wilder's method).

NOTE: Different from wilder_rsi which uses exponential smoothing.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Number of periods for RSI calculation |

**Returns:** pl.DataFrame: The input data with a new column 'rsi_sma_{period}'

### `limen.indicators.sar`

<!-- reference: api-limen-indicators-sar source=limen/indicators/sar.py region=signature:sar role=signature -->
```python
def sar(data: pl.DataFrame, high_col: str='high', low_col: str='low', acceleration: float=0.02, maximum: float=0.2) -> pl.DataFrame:
    ...
```

Compute Parabolic SAR.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high/low columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| acceleration (`float`) | Acceleration factor |
| maximum (`float`) | Maximum acceleration factor |

**Returns:** pl.DataFrame: The input data with a new column 'sar'

### `limen.indicators.sarext`

<!-- reference: api-limen-indicators-sarext source=limen/indicators/sarext.py region=signature:sarext role=signature -->
```python
def sarext(data: pl.DataFrame, high_col: str='high', low_col: str='low', start_value: float=0.0, offset_on_reverse: float=0.0, acceleration_init_long: float=0.02, acceleration_long: float=0.02, acceleration_max_long: float=0.2, acceleration_init_short: float=0.02, acceleration_short: float=0.02, acceleration_max_short: float=0.2) -> pl.DataFrame:
    ...
```

Compute Parabolic SAR - Extended (SAREXT).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high/low columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| start_value (`float`) | Start value and direction |
| offset_on_reverse (`float`) | Percent offset on reversal |
| acceleration_init_long (`float`) | Initial AF for long |
| acceleration_long (`float`) | AF increment for long |
| acceleration_max_long (`float`) | AF max for long |
| acceleration_init_short (`float`) | Initial AF for short |
| acceleration_short (`float`) | AF increment for short |
| acceleration_max_short (`float`) | AF max for short |

**Returns:** pl.DataFrame: The input data with a new column 'sarext'

### `limen.indicators.sma`

<!-- reference: api-limen-indicators-sma source=limen/indicators/sma.py region=signature:sma role=signature -->
```python
def sma(data: pl.DataFrame, price_col: str='close', period: int=30, column: str | None=None) -> pl.DataFrame:
    ...
```

Compute Simple Moving Average (SMA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |
| column (`str \| None`) | Backward-compatible alias for price_col |

**Returns:** pl.DataFrame: Input data with a new column 'sma_{period}'.

Also includes '{price_col}_sma_{period}' as a compatibility alias.

### `limen.indicators.sma_deviation_std`

<!-- reference: api-limen-indicators-sma-deviation-std source=limen/indicators/sma_deviation_std.py region=signature:sma_deviation_std role=signature -->
```python
def sma_deviation_std(data: pl.DataFrame, period: int=30, window: int=30) -> pl.DataFrame:
    ...
```

Compute rolling standard deviation of the detrended price relative to SMA(period).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Period for the SMA |
| window (`int`) | Rolling window for std of (close - SMA(period)) |

**Returns:** pl.DataFrame: The input data with a new column named using the pattern 'sma{period}_dev_std'

### `limen.indicators.stddev`

<!-- reference: api-limen-indicators-stddev source=limen/indicators/stddev.py region=signature:stddev role=signature -->
```python
def stddev(data: pl.DataFrame, price_col: str='close', period: int=5, nb_dev: float=1.0) -> pl.DataFrame:
    ...
```

Compute Standard Deviation.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |
| nb_dev (`float`) | Number of deviations to scale the output |

**Returns:** pl.DataFrame: The input data with a new column 'stddev_{period}_{nb_dev:g}'

### `limen.indicators.stoch`

<!-- reference: api-limen-indicators-stoch source=limen/indicators/stoch.py region=signature:stoch role=signature -->
```python
def stoch(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', fastk_period: int=5, slowk_period: int=3, slowk_ma_type: int=0, slowd_period: int=3, slowd_ma_type: int=0) -> pl.DataFrame:
    ...
```

Compute Stochastic Oscillator (TA_STOCH): slow %K and slow %D.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high/low/close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| fastk_period (`int`) | Time period for Fast-K (1..100000) |
| slowk_period (`int`) | Smoothing period for Slow-K (1..100000) |
| slowk_ma_type (`int`) | MA type for Slow-K (0..8) |
| slowd_period (`int`) | Smoothing period for Slow-D (1..100000) |
| slowd_ma_type (`int`) | MA type for Slow-D (0..8) |

**Returns:** pl.DataFrame: The input data with 'stoch_slowk' and 'stoch_slowd'

### `limen.indicators.stochastic_oscillator`

<!-- reference: api-limen-indicators-stochastic-oscillator source=limen/indicators/stochastic_oscillator.py region=signature:stochastic_oscillator role=signature -->
```python
def stochastic_oscillator(df: pl.DataFrame, window_k: int=14, window_d: int=3) -> pl.DataFrame:
    ...
```

Compute Stochastic Oscillator (%K and %D) using rolling highs and lows.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |
| window_k (`int`) | Number of periods for %K calculation |
| window_d (`int`) | Number of periods for %D smoothing (SMA of %K) |

**Returns:** pl.DataFrame: The input data with two new columns 'stoch_k' and 'stoch_d'

### `limen.indicators.stochf`

<!-- reference: api-limen-indicators-stochf source=limen/indicators/stochf.py region=signature:stochf role=signature -->
```python
def stochf(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', fastk_period: int=5, fastd_period: int=3, fastd_ma_type: int=0) -> pl.DataFrame:
    ...
```

Compute Fast Stochastic Oscillator (TA_STOCHF): fast %K and fast %D.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high/low/close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| fastk_period (`int`) | Time period for Fast-K (1..100000) |
| fastd_period (`int`) | Smoothing period for Fast-D (1..100000) |
| fastd_ma_type (`int`) | MA type for Fast-D (0..8) |

**Returns:** pl.DataFrame: The input data with 'stochf_fastk' and 'stochf_fastd'

### `limen.indicators.stochrsi`

<!-- reference: api-limen-indicators-stochrsi source=limen/indicators/stochrsi.py region=signature:stochrsi role=signature -->
```python
def stochrsi(data: pl.DataFrame, price_col: str='close', period: int=14, fastk_period: int=5, fastd_period: int=3, fastd_ma_type: int=0) -> pl.DataFrame:
    ...
```

Compute Stochastic RSI (TA_STOCHRSI): fast %K and fast %D on RSI values.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | RSI period (2..100000) |
| fastk_period (`int`) | Time period for Fast-K (1..100000) |
| fastd_period (`int`) | Smoothing period for Fast-D (1..100000) |
| fastd_ma_type (`int`) | MA type for Fast-D (0..8) |

**Returns:** pl.DataFrame: The input data with 'stochrsi_fastk' and 'stochrsi_fastd'

### `limen.indicators.t3`

<!-- reference: api-limen-indicators-t3 source=limen/indicators/t3.py region=signature:t3 role=signature -->
```python
def t3(data: pl.DataFrame, price_col: str='close', period: int=5, vfactor: float=0.7) -> pl.DataFrame:
    ...
```

Compute Triple Exponential Moving Average (T3).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |
| vfactor (`float`) | Volume factor |

**Returns:** pl.DataFrame: The input data with a new column 't3_{period}_{vfactor}'

### `limen.indicators.tema`

<!-- reference: api-limen-indicators-tema source=limen/indicators/tema.py region=signature:tema role=signature -->
```python
def tema(data: pl.DataFrame, price_col: str='close', period: int=30) -> pl.DataFrame:
    ...
```

Compute Triple Exponential Moving Average (TEMA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'tema_{period}'

### `limen.indicators.trange`

<!-- reference: api-limen-indicators-trange source=limen/indicators/trange.py region=signature:trange role=signature -->
```python
def trange(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute True Range (TRANGE).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high/low/close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'trange'

### `limen.indicators.trima`

<!-- reference: api-limen-indicators-trima source=limen/indicators/trima.py region=signature:trima role=signature -->
```python
def trima(data: pl.DataFrame, price_col: str='close', period: int=30) -> pl.DataFrame:
    ...
```

Compute Triangular Moving Average (TRIMA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'trima_{period}'

### `limen.indicators.trix`

<!-- reference: api-limen-indicators-trix source=limen/indicators/trix.py region=signature:trix role=signature -->
```python
def trix(data: pl.DataFrame, price_col: str='close', period: int=30) -> pl.DataFrame:
    ...
```

Compute TRIX: 1-day ROC of a triple-smoothed EMA.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods (1..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'trix_{period}'

### `limen.indicators.tsf`

<!-- reference: api-limen-indicators-tsf source=limen/indicators/tsf.py region=signature:tsf role=signature -->
```python
def tsf(data: pl.DataFrame, price_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Time Series Forecast (TSF).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'tsf_{period}'

### `limen.indicators.typprice`

<!-- reference: api-limen-indicators-typprice source=limen/indicators/typprice.py region=signature:typprice role=signature -->
```python
def typprice(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute Typical Price.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high, low, and close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'typprice'

### `limen.indicators.ultosc`

<!-- reference: api-limen-indicators-ultosc source=limen/indicators/ultosc.py region=signature:ultosc role=signature -->
```python
def ultosc(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', period1: int=7, period2: int=14, period3: int=28) -> pl.DataFrame:
    ...
```

Compute Ultimate Oscillator (ULTOSC).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high/low/close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| period1 (`int`) | Number of bars for 1st period (1..100000) |
| period2 (`int`) | Number of bars for 2nd period (1..100000) |
| period3 (`int`) | Number of bars for 3rd period (1..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'ultosc_{period1}_{period2}_{period3}'

### `limen.indicators.var`

<!-- reference: api-limen-indicators-var source=limen/indicators/var.py region=signature:var role=signature -->
```python
def var(data: pl.DataFrame, price_col: str='close', period: int=5, nb_dev: float=1.0) -> pl.DataFrame:
    ...
```

Compute Variance.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |
| nb_dev (`float`) | Kept for TA-Lib compatibility; does not affect VAR output |

**Returns:** pl.DataFrame: The input data with a new column 'var_{period}_{nb_dev:g}'

### `limen.indicators.wclprice`

<!-- reference: api-limen-indicators-wclprice source=limen/indicators/wclprice.py region=signature:wclprice role=signature -->
```python
def wclprice(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute Weighted Close Price.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high, low, and close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'wclprice'

### `limen.indicators.wilder_rsi`

<!-- reference: api-limen-indicators-wilder-rsi source=limen/indicators/wilder_rsi.py region=signature:wilder_rsi role=signature -->
```python
def wilder_rsi(data: pl.DataFrame, period: int=14) -> pl.DataFrame:
    ...
```

Compute Wilder's RSI using canonical Wilder smoothing.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Number of periods for RSI calculation |

**Returns:** pl.DataFrame: The input data with a new column 'wilder_rsi_{period}'

### `limen.indicators.willr`

<!-- reference: api-limen-indicators-willr source=limen/indicators/willr.py region=signature:willr role=signature -->
```python
def willr(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', period: int=14) -> pl.DataFrame:
    ...
```

Compute Williams' %R (WILLR).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with high/low/close columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| period (`int`) | Number of periods (2..100000) |

**Returns:** pl.DataFrame: The input data with a new column 'willr_{period}'

### `limen.indicators.window_return`

<!-- reference: api-limen-indicators-window-return source=limen/indicators/window_return.py region=signature:window_return role=signature -->
```python
def window_return(data: pl.DataFrame, period: int=24) -> pl.DataFrame:
    ...
```

Compute windowed return close/close.shift(period) - 1 for a given period.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Window length for the return |

**Returns:** pl.DataFrame: The input data with a new column named using the pattern 'ret_{period}'

### `limen.indicators.wma`

<!-- reference: api-limen-indicators-wma source=limen/indicators/wma.py region=signature:wma role=signature -->
```python
def wma(data: pl.DataFrame, price_col: str='close', period: int=30) -> pl.DataFrame:
    ...
```

Compute Weighted Moving Average (WMA).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with input price column |
| price_col (`str`) | Column name for input price |
| period (`int`) | Number of periods |

**Returns:** pl.DataFrame: The input data with a new column 'wma_{period}'
