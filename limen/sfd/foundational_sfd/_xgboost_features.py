import polars as pl

from limen.features.lagged_features import lag_columns
from limen.features.price_range_position import price_range_position
from limen.features.range_pct import range_pct
from limen.features.sma_ratios import sma_ratios
from limen.features.volume_ratio import volume_ratio
from limen.features.volume_regime import volume_regime
from limen.indicators.body_pct import body_pct
from limen.indicators.bollinger_bands import bollinger_bands
from limen.indicators.bollinger_position import bollinger_position
from limen.indicators.rolling_volatility import rolling_volatility
from limen.indicators.rsi import rsi
from limen.indicators.sma import sma
from limen.indicators.stochastic_oscillator import stochastic_oscillator
from limen.indicators.window_return import window_return


def _lag_new(data: pl.DataFrame, features: pl.DataFrame, lag: int) -> pl.DataFrame:
    columns = [col for col in features.collect_schema().names() if col not in data.collect_schema()]
    return lag_columns(features.fill_nan(None), columns, lag)


def _momentum(data: pl.DataFrame, return_fast: int, return_medium: int,
              return_slow: int, stochastic_k: int, stochastic_d: int,
              rsi_period: int, lag: int) -> pl.DataFrame:
    features = data
    for period in (return_fast, return_medium, return_slow):
        features = window_return(features, period)
    features = stochastic_oscillator(features, stochastic_k, stochastic_d)
    return _lag_new(data, rsi(features, period=rsi_period), lag)


def _trend(data: pl.DataFrame, price_sma_fast: int, price_sma_slow: int,
           lag: int) -> pl.DataFrame:
    features = sma_ratios(data, [price_sma_fast, price_sma_slow]).drop(
        [f'sma_{period}' for period in sorted({price_sma_fast, price_sma_slow})]
    )
    return _lag_new(data, features, lag)


def _volume(data: pl.DataFrame, volume_window: int, flow_window: int,
            volume_regime_window: int, lag: int) -> pl.DataFrame:
    features = sma(data, column='volume', period=volume_window)
    features = sma(features, column='maker_ratio', period=flow_window)
    features = sma(features, column='no_of_trades', period=volume_window)
    features = volume_regime(volume_ratio(features, volume_window), volume_regime_window)
    features = features.with_columns(
        (pl.col('volume') * pl.col('close')).alias('dollar_volume'),
        (pl.col('maker_ratio') - 0.5).alias('order_flow_imbalance'),
        (pl.col('no_of_trades') / pl.col(f'no_of_trades_sma_{volume_window}')).alias('trade_intensity'),
    )
    features = features.drop([f'sma_{period}' for period in sorted({volume_window, flow_window})])
    return _lag_new(data, features, lag)


def _volatility(data: pl.DataFrame, volatility_window: int, bollinger_window: int,
                bollinger_std: float, lag: int) -> pl.DataFrame:
    features = rolling_volatility(data, column='ret_1', window=volatility_window)
    features = bollinger_bands(features, window=bollinger_window, num_std=bollinger_std)
    return _lag_new(data, bollinger_position(features), lag)


def _structure(data: pl.DataFrame, price_range_window: int, lag: int) -> pl.DataFrame:
    return _lag_new(data, price_range_position(range_pct(body_pct(data)), price_range_window), lag)
