# Indicators

Indicators include common technical indicators, and any other non-compound signal that can be used for training models. 

## Purpose

There are hundreds of well-known indicators, but they broadly fall under eight different categories. These are shown in the table below.

| # | Heading | Core question (purpose) | Representative indicators | Notes |
|---|---------|-------------------------|---------------------------|-------|
| **1** | **Trend & Market Structure** | Where is price headed and how is market structure evolving? | Simple / Exponential Moving Averages (SMA/EMA), Ichimoku Cloud, Swing-high/low mapping, Market Profile, Parabolic SAR | – |
| **2** | **Momentum (Oscillators)** | How strong—or divergent—is the current move? | RSI, Stochastic, MACD histogram, Commodity Channel Index (CCI) | – |
| **3** | **Volatility** | How fast and how far might price swing? | Bollinger Bands, Average True Range (ATR), Historical/Realized Volatility, Keltner Channels | – |
| **4** | **Volume-Flow & Microstructure** | Is real money backing the move and how is it distributed across the order book? | On-Balance Volume (OBV), Volume-Weighted Average Price (VWAP)[^1], Volume Profile / VWAP Anchor, Cumulative Volume Delta (CVD), Order-book imbalance / iceberg detection | – |
| **5** | **On-Chain (Network Fundamentals)** | What is happening on the Bitcoin blockchain itself? | NVT & MVRV ratios, SOPR, Active Addresses, Halving countdown & S2F, Realized-Cap vs Market-Cap, HODL-wave age bands | Includes Bitcoin-specific *cycle & seasonality* and *valuation* metrics |
| **6** | **Derivatives-Market Metrics** | What are futures & options traders signaling? | Perpetual-swap funding rates, Open Interest, Options Implied Volatility (IV)[^2], Put/Call skew | – |
| **7** | **Sentiment & Flow** | How are humans (and bots) feeling and positioning? | Fear & Greed Index, Social-media volume, Exchange net inflows/outflows[^3], CEX/DEX positioning trackers, Google-Trends hits for “buy bitcoin” | – |
| **8** | **Macro & Liquidity** | What outside forces could push Bitcoin around? | DXY (U.S.-Dollar Index), Fed balance-sheet or global M2 growth, Treasury-yield curve shifts, Stable-coin supply expansion/contraction, BTC dominance, ETH/BTC ratio, Altcoin breadth metrics | Captures *relative-value / breadth* and cross-asset correlations |

## Indicators and SFDs

Read more about SFDs in: [Single File Decoder](Single-File-Decoder.md)

There can be an infinite number of distinct model architectures, but they all belong to one of 13 model families. Since model archicture and Indicators both give energy to SFDs, it is important to think about in a tightly coupled way. The below table provides an indication of how such thinking may appear.

| Model family | Most-suitable indicator buckets (of the 8) | Least-suitable indicator buckets | Notes |
|--------------|-------------------------------------------|----------------------------------|-------|
| **Linear models** | Trend & Market Structure; Macro & Liquidity; Derivatives-Market Metrics | Volume-Flow & Microstructure; Sentiment & Flow | Captures linear factor relationships cleanly; great for hedge-ratio sizing and interpretability, but underfits highly nonlinear order-flow or sentiment data. |
| **Tree-based models** (DT, RF, GBDT) | Volume-Flow & Microstructure; On-Chain; Sentiment & Flow | Volatility; Momentum (Oscillators) | Handle mixed data types & nonlinear splits without scaling, yet discard temporal order—limiting pure time-series uses such as realized-vol forecasting. |
| **Kernel / similarity methods** (SVM, GP) | Volatility; Sentiment & Flow; Trend & Market Structure | On-Chain; Macro & Liquidity | Flexible for medium-size nonlinear sets, but memory cost scales poorly with very wide macro or on-chain feature spaces. |
| **Instance-based / lazy learners** (k-NN, Loess) | Volume-Flow & Microstructure; Momentum (Oscillators) | Macro & Liquidity; On-Chain; Derivatives | Local-pattern matching excels at tick-level motifs yet struggles with sparse, high-dimensional macro or chain data. |
| **Neural networks** (RNN, CNN, Transformer) | Sentiment & Flow; Volume-Flow & Microstructure; Trend & Market Structure; Volatility; On-Chain | Macro & Liquidity (often small-N) | State-of-the-art for text, sequential price, and high-dimensional chain signals—data-hungry and harder to interpret. |
| **Probabilistic / graphical models** (HMM, Bayesian nets) | Volatility; Trend & Market Structure; Macro & Liquidity | Sentiment & Flow; Volume-Flow & Microstructure | Provide regime-switching and explicit uncertainty; less competitive on NLP or micro-tick speed. |
| **Ensembles / meta-learners** (Stacking, Bagging, Boosting) | Macro & Liquidity; On-Chain; Sentiment & Flow; Trend & Market Structure | — (broadly applicable) | Combine weak/heterogeneous learners into robust forecasts—often leaderboard winners across data silos. |
| **Rule & symbolic learners** (Genetic Prog., RIPPER) | Trend & Market Structure; Volume-Flow & Microstructure | Sentiment & Flow; On-Chain | Yield human-readable trading rules; prone to underfitting noisy, high-dimensional sentiment or chain features. |
| **Clustering & latent-structure models** (k-means, HDBSCAN) | On-Chain; Sentiment & Flow; Macro & Liquidity | Trend & Market Structure; Volume-Flow & Microstructure | Best for discovering regimes, cohorts, or hidden market states rather than direct signal forecasting. |
| **Dimensionality-reduction models** (PCA, UMAP, Autoencoders) | Macro & Liquidity; Sentiment & Flow; On-Chain | Trend & Market Structure; Momentum (Oscillators) | Serve mainly as preprocessing—taming collinearity and noise before downstream models. |
| **Reinforcement-learning agents** | Volume-Flow & Microstructure; Derivatives-Market Metrics; Trend & Market Structure | On-Chain; Macro & Liquidity; Sentiment & Flow | Suited to sequential decision-making (execution, allocation) where environment feedback is explicit; needs well-simulated microstructure data. |

## `limen.indicators`

### `ad`

Compute Chaikin Accumulation/Distribution (A/D) line.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Klines dataset with high/low/close/volume columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `volume_col` | `str` | Column name for volume |

#### Returns

`pl.DataFrame: The input data with a new column 'ad'`

### `adosc`

Compute Chaikin A/D Oscillator (ADOSC).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Klines dataset with high/low/close/volume columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `volume_col` | `str` | Column name for volume |
| `fast_period` | `int` | Fast EMA period |
| `slow_period` | `int` | Slow EMA period |

#### Returns

`pl.DataFrame: The input data with a new column 'adosc_{fast_period}_{slow_period}'`

### `apo`

Compute Absolute Price Oscillator (APO).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `fast_period` | `int` | Number of periods for fast MA (2..100000) |
| `slow_period` | `int` | Number of periods for slow MA (2..100000) |
| `ma_type` | `int` | TA-Lib MA type (0..8) |

#### Returns

`pl.DataFrame: The input data with a new column 'apo_{fast_period}_{slow_period}_{ma_type}'`

### `atr`

Compute Average True Range (ATR).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Klines dataset with high/low/close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `period` | `int` | Number of periods for ATR calculation |

#### Returns

`pl.DataFrame: The input data with a new column 'atr_{period}'`

### `avgprice`

Compute Average Price.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'avgprice'`

### `bbands`

Compute Bollinger Bands (upper/middle/lower).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Rolling window length |
| `nb_dev_up` | `float` | Upper-band deviation multiplier |
| `nb_dev_dn` | `float` | Lower-band deviation multiplier |
| `ma_type` | `int` | TA-Lib MA type |

#### Returns

`pl.DataFrame: Input data with 'bbands_upper', 'bbands_middle', 'bbands_lower'`

### `bop`

Compute Balance of Power (BOP).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'bop'`

### `bollinger_position`

Compute price position within the Bollinger band range.

Returns `0` at the lower band, `1` at the upper band, and `0.5` at the midpoint.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pl.DataFrame` | Klines dataset with `close`, `bb_upper`, and `bb_lower` columns |

#### Returns

`pl.DataFrame`: The input data with a new column `bollinger_position`

### `cci`

Compute Commodity Channel Index (CCI).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high/low/close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `period` | `int` | Number of periods (2..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'cci_{period}'`

### `cdl2crows`

Compute Two Crows candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdl2crows'`

### `cdl3blackcrows`

Compute Three Black Crows candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdl3blackcrows'`

### `cdl3inside`

Compute Three Inside Up/Down candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdl3inside'`

### `cdl3linestrike`

Compute Three-Line Strike candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdl3linestrike'`

### `cdl3starsinsouth`

Compute Three Stars In The South candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdl3starsinsouth'`

### `cdl3whitesoldiers`

Compute Three Advancing White Soldiers candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdl3whitesoldiers'`

### `cdlabandonedbaby`

Compute Abandoned Baby candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `penetration` | `float` | Percentage of penetration of the 3rd candle into the 1st real body |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlabandonedbaby'`

### `cdladvancedblock`

Compute Advance Block candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdladvancedblock'`

### `cdlbelthold`

Compute Belt-hold candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlbelthold'`

### `cdlclosingmarubozu`

Compute Closing Marubozu candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlclosingmarubozu'`

### `cdlconcealbabyswall`

Compute Concealing Baby Swallow candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlconcealbabyswall'`

### `cdlcounterattack`

Compute Counterattack candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlcounterattack'`

### `cdldarkcloudcover`

Compute Dark Cloud Cover candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `penetration` | `float` | Percentage of penetration of the 2nd candle into the 1st real body |

#### Returns

`pl.DataFrame: The input data with a new column 'cdldarkcloudcover'`

### `cdldragonflydoji`

Compute Dragonfly Doji candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdldragonflydoji'`

### `cdlengulfing`

Compute Engulfing candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlengulfing'`

### `cdlgravestonedoji`

Compute Gravestone Doji candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlgravestonedoji'`

### `cdlhammer`

Compute Hammer candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlhammer'`

### `cdlhangingman`

Compute Hanging Man candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlhangingman'`

### `cdlharami`

Compute Harami candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlharami'`

### `cdlharamicross`

Compute Harami Cross candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlharamicross'`

### `cdlhighwave`

Compute High-Wave Candle pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlhighwave'`

### `cdlhikkake`

Compute Hikkake candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlhikkake'`

### `cdlhikkakemod`

Compute Modified Hikkake candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlhikkakemod'`

### `cdlhomingpigeon`

Compute Homing Pigeon candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlhomingpigeon'`

### `cdlidentical3crows`

Compute Identical Three Crows candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlidentical3crows'`

### `cdlinvertedhammer`

Compute Inverted Hammer candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlinvertedhammer'`

### `cdlladderbottom`

Compute Ladder Bottom candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlladderbottom'`

### `cdllongleggeddoji`

Compute Long Legged Doji candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdllongleggeddoji'`

### `cdllongline`

Compute Long Line Candle pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdllongline'`

### `cdlmarubozu`

Compute Marubozu candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlmarubozu'`

### `cdlmatchinglow`

Compute Matching Low candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlmatchinglow'`

### `cdlmathold`

Compute Mat Hold candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `penetration` | `float` | Maximum percentage penetration of reaction days into the first white body |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlmathold'`

### `cdlonneck`

Compute On-Neck candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlonneck'`

### `cdlpiercing`

Compute Piercing candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlpiercing'`

### `cdlrickshawman`

Compute Rickshaw Man candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlrickshawman'`

### `cdlrisefall3methods`

Compute Rising/Falling Three Methods candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlrisefall3methods'`

### `cdlseparatinglines`

Compute Separating Lines candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlseparatinglines'`

### `cdlshootingstar`

Compute Shooting Star candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlshootingstar'`

### `cdlshortline`

Compute Short Line Candle pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlshortline'`

### `cdlspinningtop`

Compute Spinning Top candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlspinningtop'`

### `cdlstalledpattern`

Compute Stalled Pattern candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlstalledpattern'`

### `cdlsticksandwich`

Compute Stick Sandwich candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlsticksandwich'`

### `cdltakuri`

Compute Takuri (Dragonfly Doji with very long lower shadow) candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdltakuri'`

### `cdlthrusting`

Compute Thrusting candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlthrusting'`

### `cdltristar`

Compute Tristar candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdltristar'`

### `cdlunique3river`

Compute Unique 3 River candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'cdlunique3river'`

### `cmo`

Compute Chande Momentum Oscillator (CMO).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (2..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'cmo_{period}'`

### `coldoji`

Compute Doji candlestick pattern.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with OHLC columns |
| `open_col` | `str` | Column name for open prices |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'coldoji'`

### `dema`

Compute Double Exponential Moving Average (DEMA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'dema_{period}'`

### `ema`

Compute Exponential Moving Average (EMA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'ema_{period}'`

### `ht_dcperiod`

Compute Hilbert Transform - Dominant Cycle Period.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |

#### Returns

`pl.DataFrame: The input data with a new column 'ht_dcperiod'`

### `ht_dcphase`

Compute Hilbert Transform - Dominant Cycle Phase.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |

#### Returns

`pl.DataFrame: The input data with a new column 'ht_dcphase'`

### `ht_phasor`

Compute Hilbert Transform - Phasor Components.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |

#### Returns

`pl.DataFrame: The input data with new columns 'ht_phasor_inphase' and 'ht_phasor_quadrature'`

### `ht_sine`

Compute Hilbert Transform - SineWave.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |

#### Returns

`pl.DataFrame: The input data with new columns 'ht_sine' and 'ht_sine_lead'`

### `ht_trendline`

Compute Hilbert Transform - Instantaneous Trendline.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |

#### Returns

`pl.DataFrame: The input data with a new column 'ht_trendline'`

### `ht_trendmode`

Compute Hilbert Transform - Trend vs Cycle Mode.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |

#### Returns

`pl.DataFrame: The input data with a new column 'ht_trendmode'`

### `kama`

Compute Kaufman Adaptive Moving Average (KAMA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'kama_{period}'`

### `linearreg`

Compute Linear Regression.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'linearreg_{period}'`

### `linearreg_angle`

Compute Linear Regression Angle.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'linearreg_angle_{period}'`

### `linearreg_intercept`

Compute Linear Regression Intercept.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'linearreg_intercept_{period}'`

### `linearreg_slope`

Compute Linear Regression Slope.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'linearreg_slope_{period}'`

### `ma`

Compute Moving Average with selectable MA type (TA-Lib MA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |
| `ma_type` | `int` | TA-Lib MA type |

#### Returns

`pl.DataFrame: The input data with a new column 'ma_{period}_{ma_type}'`

### `macd`

Compute Moving Average Convergence/Divergence (MACD).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `fast_period` | `int` | Number of periods for fast EMA (2..100000) |
| `slow_period` | `int` | Number of periods for slow EMA (2..100000) |
| `signal_period` | `int` | Number of periods for signal EMA (1..100000) |

#### Returns

`pl.DataFrame: The input data with columns 'macd', 'macd_signal', 'macd_hist'`

### `macdext`

Compute MACD with controllable MA types (MACDEXT).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `fast_period` | `int` | Number of periods for fast MA (2..100000) |
| `fast_ma_type` | `int` | TA-Lib MA type for fast MA (0..8) |
| `slow_period` | `int` | Number of periods for slow MA (2..100000) |
| `slow_ma_type` | `int` | TA-Lib MA type for slow MA (0..8) |
| `signal_period` | `int` | Number of periods for signal MA (1..100000) |
| `signal_ma_type` | `int` | TA-Lib MA type for signal MA (0..8) |

#### Returns

`pl.DataFrame: The input data with columns 'macdext', 'macdext_signal', 'macdext_hist'`

### `macdfix`

Compute MACD Fix 12/26 (MACDFIX).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `signal_period` | `int` | Number of periods for signal EMA (1..100000) |

#### Returns

`pl.DataFrame: The input data with columns 'macdfix', 'macdfix_signal', 'macdfix_hist'`

### `mama`

Compute MESA Adaptive Moving Average (MAMA) and Following Adaptive MA (FAMA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `fast_limit` | `float` | Upper adaptive limit |
| `slow_limit` | `float` | Lower adaptive limit |

#### Returns

`pl.DataFrame: The input data with new columns 'mama' and 'fama'`

### `medprice`

Compute Median Price.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high and low columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |

#### Returns

`pl.DataFrame: The input data with a new column 'medprice'`

### `mfi`

Compute Money Flow Index (MFI).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Klines dataset with high/low/close/volume columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `volume_col` | `str` | Column name for volume |
| `period` | `int` | MFI period |

#### Returns

`pl.DataFrame: The input data with a new column 'mfi_{period}'`

### `midpoint`

Compute MidPoint over period.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (2..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'midpoint_{period}'`

### `midprice`

Compute Midpoint Price over period.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high and low columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `period` | `int` | Number of periods (2..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'midprice_{period}'`

### `mom`

Compute Momentum (MOM).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (1..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'mom_{period}'`

### `natr`

Compute Normalized Average True Range (NATR).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Klines dataset with high/low/close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `period` | `int` | Number of periods for NATR calculation |

#### Returns

`pl.DataFrame: The input data with a new column 'natr_{period}'`

### `obv`

Compute On-Balance Volume (OBV).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with price and volume columns |
| `price_col` | `str` | Column name for price series |
| `volume_col` | `str` | Column name for volume series |

#### Returns

`pl.DataFrame: The input data with a new column 'obv'`

### `ppo`

Compute Percentage Price Oscillator (PPO).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `fast_period` | `int` | Number of periods for fast MA (2..100000) |
| `slow_period` | `int` | Number of periods for slow MA (2..100000) |
| `ma_type` | `int` | TA-Lib MA type (0..8) |

#### Returns

`pl.DataFrame: The input data with a new column 'ppo_{fast_period}_{slow_period}_{ma_type}'`

### `roc`

Compute Rate of Change (ROC): ((price / prev_price) - 1) * 100.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (1..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'roc_{period}'`

### `rocp`

Compute Rate of Change Percentage (ROCP): (price - prev_price) / prev_price.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (1..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'rocp_{period}'`

### `rocr`

Compute Rate of Change Ratio (ROCR): price / prev_price.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (1..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'rocr_{period}'`

### `rocr100`

Compute Rate of Change Ratio 100 scale (ROCR100): (price / prev_price) * 100.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (1..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'rocr100_{period}'`

### `rsi`

Compute Relative Strength Index (RSI) using Wilder smoothing.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (2..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'rsi_{period}'`

### `sar`

Compute Parabolic SAR.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high/low columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `acceleration` | `float` | Acceleration factor |
| `maximum` | `float` | Maximum acceleration factor |

#### Returns

`pl.DataFrame: The input data with a new column 'sar'`

### `sarext`

Compute Parabolic SAR - Extended (SAREXT).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high/low columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `start_value` | `float` | Start value and direction |
| `offset_on_reverse` | `float` | Percent offset on reversal |
| `acceleration_init_long` | `float` | Initial AF for long |
| `acceleration_long` | `float` | AF increment for long |
| `acceleration_max_long` | `float` | AF max for long |
| `acceleration_init_short` | `float` | Initial AF for short |
| `acceleration_short` | `float` | AF increment for short |
| `acceleration_max_short` | `float` | AF max for short |

#### Returns

`pl.DataFrame: The input data with a new column 'sarext'`

### `sma`

Compute Simple Moving Average (SMA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |
| `column` | `str \| None` | Backward-compatible alias for price_col |

#### Returns

`pl.DataFrame: Input data with a new column 'sma_{period}'. Also includes '{price_col}_sma_{period}' as a compatibility alias.`

### `stddev`

Compute Standard Deviation.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |
| `nb_dev` | `float` | Number of deviations to scale the output |

#### Returns

`pl.DataFrame: The input data with a new column 'stddev_{period}_{nb_dev:g}'`

### `stoch`

Compute Stochastic Oscillator (TA_STOCH): slow %K and slow %D.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high/low/close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `fastk_period` | `int` | Time period for Fast-K (1..100000) |
| `slowk_period` | `int` | Smoothing period for Slow-K (1..100000) |
| `slowk_ma_type` | `int` | MA type for Slow-K (0..8) |
| `slowd_period` | `int` | Smoothing period for Slow-D (1..100000) |
| `slowd_ma_type` | `int` | MA type for Slow-D (0..8) |

#### Returns

`pl.DataFrame: The input data with 'stoch_slowk' and 'stoch_slowd'`

### `stochf`

Compute Fast Stochastic Oscillator (TA_STOCHF): fast %K and fast %D.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high/low/close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `fastk_period` | `int` | Time period for Fast-K (1..100000) |
| `fastd_period` | `int` | Smoothing period for Fast-D (1..100000) |
| `fastd_ma_type` | `int` | MA type for Fast-D (0..8) |

#### Returns

`pl.DataFrame: The input data with 'stochf_fastk' and 'stochf_fastd'`

### `stochrsi`

Compute Stochastic RSI (TA_STOCHRSI): fast %K and fast %D on RSI values.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | RSI period (2..100000) |
| `fastk_period` | `int` | Time period for Fast-K (1..100000) |
| `fastd_period` | `int` | Smoothing period for Fast-D (1..100000) |
| `fastd_ma_type` | `int` | MA type for Fast-D (0..8) |

#### Returns

`pl.DataFrame: The input data with 'stochrsi_fastk' and 'stochrsi_fastd'`

### `t3`

Compute Triple Exponential Moving Average (T3).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |
| `vfactor` | `float` | Volume factor |

#### Returns

`pl.DataFrame: The input data with a new column 't3_{period}_{vfactor}'`

### `tema`

Compute Triple Exponential Moving Average (TEMA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'tema_{period}'`

### `trange`

Compute True Range (TRANGE).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Klines dataset with high/low/close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'trange'`

### `trima`

Compute Triangular Moving Average (TRIMA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'trima_{period}'`

### `trix`

Compute TRIX: 1-day ROC of a triple-smoothed EMA.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods (1..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'trix_{period}'`

### `tsf`

Compute Time Series Forecast (TSF).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'tsf_{period}'`

### `typprice`

Compute Typical Price.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high, low, and close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'typprice'`

### `ultosc`

Compute Ultimate Oscillator (ULTOSC).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high/low/close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `period1` | `int` | Number of bars for 1st period (1..100000) |
| `period2` | `int` | Number of bars for 2nd period (1..100000) |
| `period3` | `int` | Number of bars for 3rd period (1..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'ultosc_{period1}_{period2}_{period3}'`

### `var`

Compute Variance.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |
| `nb_dev` | `float` | Kept for TA-Lib compatibility; does not affect VAR output |

#### Returns

`pl.DataFrame: The input data with a new column 'var_{period}_{nb_dev:g}'`

### `wclprice`

Compute Weighted Close Price.

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high, low, and close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |

#### Returns

`pl.DataFrame: The input data with a new column 'wclprice'`

### `willr`

Compute Williams' %R (WILLR).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with high/low/close columns |
| `high_col` | `str` | Column name for high prices |
| `low_col` | `str` | Column name for low prices |
| `close_col` | `str` | Column name for close prices |
| `period` | `int` | Number of periods (2..100000) |

#### Returns

`pl.DataFrame: The input data with a new column 'willr_{period}'`

### `wma`

Compute Weighted Moving Average (WMA).

#### Args

| Parameter | Type | Description |
|---|---|---|
| `data` | `pl.DataFrame` | Dataset with input price column |
| `price_col` | `str` | Column name for input price |
| `period` | `int` | Number of periods |

#### Returns

`pl.DataFrame: The input data with a new column 'wma_{period}'`

### `avgprice`

Compute Average Price (AVGPRICE) indicator.

Equivalent to TA-Lib AVGPRICE: (open + high + low + close) / 4.

#### Args

| Parameter   | Type            | Description                              |
|-------------|-----------------|------------------------------------------|
| `data`      | `pl.DataFrame`  | Klines dataset with OHLC columns         |
| `open_col`  | `str`           | Column name for open prices              |
| `high_col`  | `str`           | Column name for high prices              |
| `low_col`   | `str`           | Column name for low prices               |
| `close_col` | `str`           | Column name for close prices             |

#### Returns

`pl.DataFrame`: The input data with a new column 'avgprice'

### `medprice`

Compute Median Price (MEDPRICE) indicator.

Equivalent to TA-Lib MEDPRICE: (high + low) / 2.

#### Args

| Parameter  | Type            | Description                                    |
|------------|-----------------|------------------------------------------------|
| `data`     | `pl.DataFrame`  | Klines dataset with 'high' and 'low' columns  |
| `high_col` | `str`           | Column name for high prices                    |
| `low_col`  | `str`           | Column name for low prices                     |

#### Returns

`pl.DataFrame`: The input data with a new column 'medprice'

### `midprice`

Compute Midpoint Price Over Period (MIDPRICE) indicator.

Equivalent to TA-Lib MIDPRICE: (rolling_max(high, period) + rolling_min(low, period)) / 2.

#### Args

| Parameter  | Type            | Description                                    |
|------------|-----------------|------------------------------------------------|
| `data`     | `pl.DataFrame`  | Klines dataset with 'high' and 'low' columns  |
| `high_col` | `str`           | Column name for high prices                    |
| `low_col`  | `str`           | Column name for low prices                     |
| `period`   | `int`           | Number of periods for the rolling window       |

#### Returns

`pl.DataFrame`: The input data with a new column 'midprice_{period}'

### `midpoint`

Compute rolling midpoint of a single column over a window.

Equivalent to TA-Lib MIDPOINT on the chosen series.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pl.DataFrame` | Klines dataset with the target column |
| `col` | `str` | Column name to compute midpoint on |
| `period` | `int` | Number of periods for the rolling window |

#### Returns

`pl.DataFrame`: The input data with a new column '{col}_midpoint_{period}' (for the default `col='close'`, this is 'close_midpoint_{period}')

### `typprice`

Compute Typical Price (TYPPRICE) indicator.

Equivalent to TA-Lib TYPPRICE: (high + low + close) / 3.

#### Args

| Parameter   | Type            | Description                                          |
|-------------|-----------------|------------------------------------------------------|
| `data`      | `pl.DataFrame`  | Klines dataset with 'high', 'low', 'close' columns  |
| `high_col`  | `str`           | Column name for high prices                          |
| `low_col`   | `str`           | Column name for low prices                           |
| `close_col` | `str`           | Column name for close prices                         |

#### Returns

`pl.DataFrame`: The input data with a new column 'typprice'

### `wclprice`

Compute Weighted Close Price (WCLPRICE) indicator.

Equivalent to TA-Lib WCLPRICE: (high + low + 2 * close) / 4.

#### Args

| Parameter   | Type            | Description                                          |
|-------------|-----------------|------------------------------------------------------|
| `data`      | `pl.DataFrame`  | Klines dataset with 'high', 'low', 'close' columns  |
| `high_col`  | `str`           | Column name for high prices                          |
| `low_col`   | `str`           | Column name for low prices                           |
| `close_col` | `str`           | Column name for close prices                         |

#### Returns

`pl.DataFrame`: The input data with a new column 'wclprice'

### `var`

Compute Variance (VAR) over a rolling period.

Equivalent to TA-Lib VAR: rolling sample variance (ddof=1) over `period` bars.

#### Args

| Parameter | Type            | Description                                  |
|-----------|-----------------|----------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with price column             |
| `col`     | `str`           | Column name for price data                   |
| `period`  | `int`           | Number of periods for the rolling window     |

#### Returns

`pl.DataFrame`: The input data with a new column 'var_{period}'

### `linearreg`

Compute Linear Regression value (LINEARREG) indicator.

Equivalent to TA-Lib LINEARREG: the value of the least-squares regression line at the last point of each `period`-bar window. Uses a vectorised closed-form OLS formula with time indices [0, 1, ..., period-1].

#### Args

| Parameter | Type            | Description                                  |
|-----------|-----------------|----------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with price column             |
| `col`     | `str`           | Column name for price data                   |
| `period`  | `int`           | Number of periods for the rolling window     |

#### Returns

`pl.DataFrame`: The input data with a new column 'linearreg_{period}'

### `linearreg_slope`

Compute Linear Regression Slope (LINEARREG_SLOPE) indicator.

Equivalent to TA-Lib LINEARREG_SLOPE: the slope of the least-squares regression line fitted to each `period`-bar window. Uses a vectorised closed-form OLS formula.

#### Args

| Parameter | Type            | Description                                  |
|-----------|-----------------|----------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with price column             |
| `col`     | `str`           | Column name for price data                   |
| `period`  | `int`           | Number of periods for the rolling window     |

#### Returns

`pl.DataFrame`: The input data with a new column 'linearreg_slope_{period}'

### `linearreg_intercept`

Compute Linear Regression Intercept (LINEARREG_INTERCEPT) indicator.

Equivalent to TA-Lib LINEARREG_INTERCEPT: the y-intercept of the least-squares regression line fitted to each `period`-bar window. Uses a vectorised closed-form OLS formula.

#### Args

| Parameter | Type            | Description                                  |
|-----------|-----------------|----------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with price column             |
| `col`     | `str`           | Column name for price data                   |
| `period`  | `int`           | Number of periods for the rolling window     |

#### Returns

`pl.DataFrame`: The input data with a new column 'linearreg_intercept_{period}'

### `linearreg_angle`

Compute Linear Regression Angle (LINEARREG_ANGLE) indicator.

Equivalent to TA-Lib LINEARREG_ANGLE: the angle in degrees of the slope of the least-squares regression line fitted to each `period`-bar window. Computed as atan(slope) * (180 / pi).

#### Args

| Parameter | Type            | Description                                  |
|-----------|-----------------|----------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with price column             |
| `col`     | `str`           | Column name for price data                   |
| `period`  | `int`           | Number of periods for the rolling window     |

#### Returns

`pl.DataFrame`: The input data with a new column 'linearreg_angle_{period}'

---

[^1]: **VWAP** is both a trend-following anchor and a volume-weighted flow metric. We park it in **Volume-Flow & Microstructure** to keep all order-flow tools together, but many chartists also treat it as a trend indicator.
[^2]: **Implied Volatility (IV)** lives in **Derivatives** because the data source is options markets—even though it doubles as a forward-looking volatility gauge.
[^3]: **Exchange flows** originate on-chain, yet we place them under **Sentiment & Flow** because traders mainly interpret net inflows/outflows as fear-vs-complacency signals.
