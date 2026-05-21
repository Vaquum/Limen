import math
from collections.abc import Sequence

import polars as pl


PARKINSON_SCALE = 4.0 * math.log(2.0)


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator == 0.0))
        .then(None)
        .otherwise(numerator / denominator)
    )


def _price_range(high_col: str = 'high', low_col: str = 'low') -> pl.Expr:
    return pl.col(high_col) - pl.col(low_col)


def _parkinson_variance(high_col: str = 'high', low_col: str = 'low') -> pl.Expr:
    return ((pl.col(high_col).log() - pl.col(low_col).log()) ** 2) / PARKINSON_SCALE


def _true_range(
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.Expr:
    previous_close = pl.col(close_col).shift(1)
    return pl.max_horizontal(
        _price_range(high_col, low_col),
        (pl.col(high_col) - previous_close).abs(),
        (pl.col(low_col) - previous_close).abs(),
    )


def wick_proportion(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    open_col: str = 'open',
    close_col: str = 'close',
    output_col: str = 'wick_proportion',
) -> pl.DataFrame:

    '''Compute rolling mean wick share of the full candle range.'''

    range_expr = _price_range(high_col, low_col)
    wick_expr = range_expr - (pl.col(close_col) - pl.col(open_col)).abs()
    return data.with_columns(
        _safe_divide(wick_expr, range_expr)
        .rolling_mean(window_size=window)
        .alias(output_col)
    )


def stochastic_k_abs(
    data: pl.DataFrame,
    window: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    output_col: str = 'stochastic_k_abs',
) -> pl.DataFrame:

    '''Compute absolute distance of stochastic %K from the center line.'''

    rolling_low = pl.col(low_col).rolling_min(window_size=window)
    rolling_high = pl.col(high_col).rolling_max(window_size=window)
    stoch_k = _safe_divide(pl.col(close_col) - rolling_low, rolling_high - rolling_low)
    return data.with_columns((stoch_k - 0.5).abs().alias(output_col))


def volatility_ratio(
    data: pl.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'volatility_ratio',
) -> pl.DataFrame:

    '''Compare short and long rolling means of Parkinson variance.'''

    variance = _parkinson_variance(high_col, low_col)
    return data.with_columns(
        _safe_divide(
            variance.rolling_mean(window_size=short_window),
            variance.rolling_mean(window_size=long_window),
        ).alias(output_col)
    )


def close_position_rolling(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    output_col: str = 'close_position_rolling',
) -> pl.DataFrame:

    '''Compute rolling mean close position inside the high-low range.'''

    position = _safe_divide(pl.col(close_col) - pl.col(low_col), _price_range(high_col, low_col))
    return data.with_columns(position.rolling_mean(window_size=window).alias(output_col))


def distance_from_ma(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
    output_col: str = 'distance_from_ma',
) -> pl.DataFrame:

    '''Compute close distance from its rolling moving average.'''

    moving_average = pl.col(close_col).rolling_mean(window_size=window)
    return data.with_columns(
        _safe_divide(pl.col(close_col) - moving_average, moving_average).alias(output_col)
    )


def close_ma_distance_atr(
    data: pl.DataFrame,
    ma_window: int = 20,
    atr_window: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    output_col: str = 'close_ma_distance_atr',
) -> pl.DataFrame:

    '''Compute close-to-SMA distance normalized by SMA-smoothed true range.'''

    moving_average = pl.col(close_col).rolling_mean(window_size=ma_window)
    atr = _true_range(high_col, low_col, close_col).rolling_mean(window_size=atr_window)
    return data.with_columns(
        _safe_divide(pl.col(close_col) - moving_average, atr).alias(output_col)
    )


def kaufman_efficiency_ratio(
    data: pl.DataFrame,
    window: int = 10,
    close_col: str = 'close',
    output_col: str = 'kaufman_efficiency_ratio',
) -> pl.DataFrame:

    '''Compute Kaufman's efficiency ratio over a rolling window.'''

    displacement = (pl.col(close_col) - pl.col(close_col).shift(window)).abs()
    path_length = (pl.col(close_col) - pl.col(close_col).shift(1)).abs().rolling_sum(window_size=window)
    return data.with_columns(_safe_divide(displacement, path_length).alias(output_col))


def return_autocorrelation(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
    output_col: str = 'return_autocorrelation',
) -> pl.DataFrame:

    '''Compute rolling autocorrelation between returns and one-bar lagged returns.'''

    return_expr = pl.col(close_col).pct_change()
    return data.with_columns(
        pl.rolling_corr(return_expr, return_expr.shift(1), window_size=window).alias(output_col)
    )


def volume_volatility_correlation(
    data: pl.DataFrame,
    window: int = 20,
    volume_col: str = 'volume',
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'volume_volatility_correlation',
) -> pl.DataFrame:

    '''Compute rolling correlation between volume and Parkinson variance.'''

    return data.with_columns(
        pl.rolling_corr(
            pl.col(volume_col),
            _parkinson_variance(high_col, low_col),
            window_size=window,
        ).alias(output_col)
    )


def return_volatility_correlation(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'return_volatility_correlation',
) -> pl.DataFrame:

    '''Compute rolling correlation between returns and Parkinson variance.'''

    return data.with_columns(
        pl.rolling_corr(
            pl.col(close_col).pct_change(),
            _parkinson_variance(high_col, low_col),
            window_size=window,
        ).alias(output_col)
    )


def downside_volatility_ratio(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
    output_col: str = 'downside_volatility_ratio',
) -> pl.DataFrame:

    '''Compute rolling downside squared-return share of total squared returns.'''

    returns = pl.col(close_col).pct_change()
    squared = returns ** 2
    downside = pl.when(returns < 0.0).then(squared).otherwise(0.0)
    return data.with_columns(
        _safe_divide(
            downside.rolling_sum(window_size=window),
            squared.rolling_sum(window_size=window),
        ).alias(output_col)
    )


def narrow_range(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'narrow_range',
) -> pl.DataFrame:

    '''Compute current range relative to the trailing maximum range.'''

    range_expr = _price_range(high_col, low_col)
    return data.with_columns(
        _safe_divide(range_expr, range_expr.rolling_max(window_size=window)).alias(output_col)
    )


def volume_to_range(
    data: pl.DataFrame,
    window: int = 20,
    volume_col: str = 'volume',
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'volume_to_range',
) -> pl.DataFrame:

    '''Compute rolling mean volume per unit of high-low range.'''

    volume_per_range = _safe_divide(pl.col(volume_col), _price_range(high_col, low_col))
    return data.with_columns(volume_per_range.rolling_mean(window_size=window).alias(output_col))


def volatility_spike(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'volatility_spike',
) -> pl.DataFrame:

    '''Compute current Parkinson variance relative to its fixed-lag value.'''

    variance = _parkinson_variance(high_col, low_col)
    return data.with_columns(_safe_divide(variance, variance.shift(window)).alias(output_col))


def parkinson_vol_of_vol(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'parkinson_vol_of_vol',
) -> pl.DataFrame:

    '''Compute rolling standard deviation of Parkinson variance.'''

    return data.with_columns(
        _parkinson_variance(high_col, low_col)
        .rolling_std(window_size=window)
        .alias(output_col)
    )


__all__: Sequence[str] = [
    'close_ma_distance_atr',
    'close_position_rolling',
    'distance_from_ma',
    'downside_volatility_ratio',
    'kaufman_efficiency_ratio',
    'narrow_range',
    'parkinson_vol_of_vol',
    'return_autocorrelation',
    'return_volatility_correlation',
    'stochastic_k_abs',
    'volatility_ratio',
    'volatility_spike',
    'volume_to_range',
    'volume_volatility_correlation',
    'wick_proportion',
]
