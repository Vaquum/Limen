from collections.abc import Sequence

import polars as pl


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator == 0.0))
        .then(None)
        .otherwise(numerator / denominator)
    )


def maker_volume_share(
    data: pl.DataFrame,
    maker_volume_col: str = 'maker_volume',
    volume_col: str = 'volume',
    output_col: str = 'maker_volume_share',
) -> pl.DataFrame:

    '''Compute maker volume as a share of total volume.'''

    return data.with_columns(
        _safe_divide(pl.col(maker_volume_col), pl.col(volume_col)).alias(output_col)
    )


def maker_liquidity_share(
    data: pl.DataFrame,
    maker_liquidity_col: str = 'maker_liquidity',
    liquidity_col: str = 'liquidity_sum',
    output_col: str = 'maker_liquidity_share',
) -> pl.DataFrame:

    '''Compute maker liquidity as a share of total liquidity.'''

    return data.with_columns(
        _safe_divide(pl.col(maker_liquidity_col), pl.col(liquidity_col)).alias(output_col)
    )


def maker_volume_ratio(
    data: pl.DataFrame,
    window: int = 20,
    maker_volume_col: str = 'maker_volume',
    volume_col: str = 'volume',
    output_col: str = 'maker_volume_ratio',
) -> pl.DataFrame:

    '''Compute the rolling mean of maker volume share.'''

    share = _safe_divide(pl.col(maker_volume_col), pl.col(volume_col))
    return data.with_columns(share.rolling_mean(window_size=window).alias(output_col))


def trade_imbalance(
    data: pl.DataFrame,
    window: int = 20,
    maker_volume_col: str = 'maker_volume',
    volume_col: str = 'volume',
    output_col: str = 'trade_imbalance',
) -> pl.DataFrame:

    '''Compute rolling maker-volume share from rolling sums.'''

    return data.with_columns(
        _safe_divide(
            pl.col(maker_volume_col).rolling_sum(window_size=window),
            pl.col(volume_col).rolling_sum(window_size=window),
        ).alias(output_col)
    )


def taker_imbalance_ratio(
    data: pl.DataFrame,
    window: int = 20,
    volume_col: str = 'volume',
    maker_ratio_col: str = 'maker_ratio',
    output_col: str = 'taker_imbalance_ratio',
) -> pl.DataFrame:

    '''Compute rolling mean absolute taker imbalance as a share of volume.'''

    imbalance = (pl.col(volume_col) * (1.0 - (2.0 * pl.col(maker_ratio_col)))).abs()
    ratio = _safe_divide(imbalance, pl.col(volume_col))
    return data.with_columns(ratio.rolling_mean(window_size=window).alias(output_col))


def trade_density(
    data: pl.DataFrame,
    window: int = 20,
    trades_col: str = 'no_of_trades',
    volume_col: str = 'volume',
    output_col: str = 'trade_density',
) -> pl.DataFrame:

    '''Compute rolling mean number of trades per unit of volume.'''

    density = _safe_divide(pl.col(trades_col), pl.col(volume_col))
    return data.with_columns(density.rolling_mean(window_size=window).alias(output_col))


def trade_size_ratio(
    data: pl.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    volume_col: str = 'volume',
    trades_col: str = 'no_of_trades',
    output_col: str = 'trade_size_ratio',
) -> pl.DataFrame:

    '''Compare short and long rolling average trade sizes.'''

    trade_size = _safe_divide(pl.col(volume_col), pl.col(trades_col))
    return data.with_columns(
        _safe_divide(
            trade_size.rolling_mean(window_size=short_window),
            trade_size.rolling_mean(window_size=long_window),
        ).alias(output_col)
    )


def liquidity_range(
    data: pl.DataFrame,
    window: int = 20,
    high_liquidity_col: str = 'high_liquidity',
    low_liquidity_col: str = 'low_liquidity',
    output_col: str = 'liquidity_range',
) -> pl.DataFrame:

    '''Compute rolling mean high-liquidity to low-liquidity ratio.'''

    ratio = _safe_divide(pl.col(high_liquidity_col), pl.col(low_liquidity_col))
    return data.with_columns(ratio.rolling_mean(window_size=window).alias(output_col))


def liquidity_drop(
    data: pl.DataFrame,
    window: int = 20,
    liquidity_col: str = 'liquidity_sum',
    output_col: str = 'liquidity_drop',
) -> pl.DataFrame:

    '''Compute current liquidity relative to liquidity `window` bars ago.'''

    return data.with_columns(
        _safe_divide(pl.col(liquidity_col), pl.col(liquidity_col).shift(window)).alias(output_col)
    )


__all__: Sequence[str] = [
    'liquidity_drop',
    'liquidity_range',
    'maker_liquidity_share',
    'maker_volume_ratio',
    'maker_volume_share',
    'taker_imbalance_ratio',
    'trade_density',
    'trade_imbalance',
    'trade_size_ratio',
]
