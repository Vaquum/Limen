# Indicators

Indicators include common technical indicators, and any other non-compound signal that can be used for training models. 

## Purpose

There are hundreds of well-known indicators, but they all fall under eight different categories. These are demonstrated in the below table.

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

## Indicators and SFMs

Read more about SFMs in: [Single File Model](Single-File-Model.md)

There can be an infinite number of distinct model architectures, but they all belong to one of 13 model families. Since model archicture and Indicators both give energy to SFMs, it is important to think about in a tightly coupled way. The below table provides an indication of how such thinking may appear.

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

## `loop.indicators`

### `atr`

Compute Average True Range (ATR) using Wilder's smoothing method.

#### Args

| Parameter   | Type            | Description                                      |
|-------------|-----------------|--------------------------------------------------|
| `data`      | `pl.DataFrame`  | Klines dataset with 'high', 'low', 'close' columns |
| `high_col`  | `str`           | Column name for high prices                      |
| `low_col`   | `str`           | Column name for low prices                       |
| `close_col` | `str`           | Column name for close prices                     |
| `period`    | `int`           | Number of periods for ATR calculation           |

#### Returns

`pl.DataFrame`: The input data with a new column 'atr_{period}'

### `body_pct`

Compute the body percentage (candle body size relative to open).

#### Args

| Parameter | Type            | Description                                      |
|-----------|-----------------|--------------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with 'open' and 'close' columns  |

#### Returns

`pl.DataFrame`: The input data with a new column 'body_pct'

### `macd`

Compute MACD (Moving Average Convergence Divergence) indicator.

#### Args

| Parameter       | Type            | Description                                           |
|-----------------|-----------------|-------------------------------------------------------|
| `data`          | `pl.DataFrame`  | Klines dataset with 'close' column                   |
| `close_col`     | `str`           | Column name for close prices                         |
| `fast_period`   | `int`           | Period for fast EMA calculation                      |
| `slow_period`   | `int`           | Period for slow EMA calculation                      |
| `signal_period` | `int`           | Period for signal line EMA calculation               |

#### Returns

`pl.DataFrame`: The input data with three columns: 'macd_{fast_period}_{slow_period}', 'macd_signal_{signal_period}', 'macd_hist'

### `ppo`

Compute Percentage Price Oscillator (PPO) indicator.

#### Args

| Parameter    | Type            | Description                                      |
|--------------|-----------------|--------------------------------------------------|
| `data`       | `pl.DataFrame`  | Klines dataset with price column                 |
| `close_col`  | `str`           | Column name for price data                       |
| `fast_period` | `int`           | Period for short EMA calculation                 |
| `slow_period`  | `int`           | Period for long EMA calculation                  |
| `signal_period` | `int`           | Period for signal line EMA calculation               |

#### Returns

`pl.DataFrame`: The input data with three columns: 'ppo_{fast_period}_{slow_period}', 'ppo_signal_{signal_period}', 'ppo_hist'

### `price_change_pct`

Compute price change percentage over a specific period.

#### Args

| Parameter | Type            | Description                            |
|-----------|-----------------|----------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with 'close' column    |
| `period`  | `int`           | Number of periods to look back         |

#### Returns

`pl.DataFrame`: The input data with a new column 'price_change_pct_{period}'

### `returns`

Compute period-over-period returns of close prices.

#### Args

| Parameter | Type            | Description                            |
|-----------|-----------------|----------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with 'close' column    |

#### Returns

`pl.DataFrame`: The input data with a new column 'returns'

### `roc`

Compute Rate of Change (ROC) indicator as percentage change.

#### Args

| Parameter | Type            | Description                                      |
|-----------|-----------------|--------------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with price column                 |
| `col`     | `str`           | Column name for price data                       |
| `period`  | `int`           | Number of periods for ROC calculation           |

#### Returns

`pl.DataFrame`: The input data with a new column 'roc_{period}'

### `rolling_volatility`

Compute rolling volatility (standard deviation) over a specified period.

#### Args

| Parameter | Type            | Description                                           |
|-----------|-----------------|-------------------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with price/returns column             |
| `column`  | `str`           | Column name to calculate volatility on (typically returns) |
| `window`  | `int`           | Number of periods for rolling window calculation     |

#### Returns

`pl.DataFrame`: The input data with a new column '{column}\_volatility_{window}'

### `rsi_sma`

Compute RSI using Simple Moving Average smoothing (not Wilder's method).

NOTE: Different from wilder_rsi which uses exponential smoothing.

#### Args

| Parameter | Type            | Description                                      |
|-----------|-----------------|--------------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with 'close' column              |
| `period`  | `int`           | Number of periods for RSI calculation           |

#### Returns

`pl.DataFrame`: The input data with a new column 'rsi_sma_{period}'

### `sma`

Compute Simple Moving Average (SMA) indicator.

#### Args

| Parameter | Type            | Description                            |
|-----------|-----------------|----------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with price column       |
| `column`  | `str`           | Column name to calculate SMA on        |
| `period`  | `int`           | Number of periods for SMA calculation  |

#### Returns

`pl.DataFrame`: The input data with a new column '{column}_sma_{period}'

### `wilder_rsi`

Compute Wilder's RSI using exponential smoothing method.

#### Args

| Parameter | Type            | Description                                      |
|-----------|-----------------|--------------------------------------------------|
| `data`    | `pl.DataFrame`  | Klines dataset with 'close' column              |
| `period`  | `int`           | Number of periods for RSI calculation           |

#### Returns

`pl.DataFrame`: The input data with a new column 'wilder_rsi_{period}'

---

[^1]: **VWAP** is both a trend-following anchor and a volume-weighted flow metric. We park it in **Volume-Flow & Microstructure** to keep all order-flow tools together, but many chartists also treat it as a trend indicator.  
[^2]: **Implied Volatility (IV)** lives in **Derivatives** because the data source is options markets—even though it doubles as a forward-looking volatility gauge.  
[^3]: **Exchange flows** originate on-chain, yet we place them under **Sentiment & Flow** because traders mainly interpret net inflows/outflows as fear-vs-complacency signals.