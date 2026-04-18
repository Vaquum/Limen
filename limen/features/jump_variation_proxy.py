import math

import polars as pl


def jump_variation_proxy(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute a rolling jump-variation proxy from realized variance and bipower variation.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods used for the rolling jump proxy
        close_col (str): Column name used for log returns

    Returns:
        pl.DataFrame: The input data with a new column 'jump_variation_proxy'
    '''

    mu1_inverse_sq = math.pi / 2.0

    return (
        data
        .with_columns((pl.col(close_col).log() - pl.col(close_col).shift(1).log()).alias('_jump_returns'))
        .with_columns([
            (pl.col('_jump_returns') ** 2).rolling_sum(window_size=window).alias('_jump_realized_variance'),
            (
                mu1_inverse_sq
                * pl.col('_jump_returns').abs()
                * pl.col('_jump_returns').shift(1).abs()
            )
            .rolling_sum(window_size=window)
            .alias('_jump_bipower_variation'),
        ])
        .with_columns(
            (
                (pl.col('_jump_realized_variance') - pl.col('_jump_bipower_variation'))
                .clip(lower_bound=0.0)
            ).alias('jump_variation_proxy')
        )
        .drop([
            '_jump_returns',
            '_jump_realized_variance',
            '_jump_bipower_variation',
        ])
    )
