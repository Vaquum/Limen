import polars as pl

EPSILON = 1e-10


def volatility_term_structure(
    data: pl.DataFrame,
    short_window: int = 12,
    medium_window: int = 48,
    long_window: int = 168,
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute a front-versus-back volatility term-structure ratio across three horizons.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        short_window (int): Short volatility horizon
        medium_window (int): Medium volatility horizon
        long_window (int): Long volatility horizon
        close_col (str): Column name used for close-to-close returns

    Returns:
        pl.DataFrame: The input data with a new column 'volatility_term_structure'
    '''

    returns = pl.col(close_col).pct_change()

    return (
        data
        .with_columns(returns.alias('_term_structure_returns'))
        .with_columns([
            pl.col('_term_structure_returns').rolling_std(window_size=short_window).alias('_term_structure_short_vol'),
            pl.col('_term_structure_returns').rolling_std(window_size=medium_window).alias('_term_structure_medium_vol'),
            pl.col('_term_structure_returns').rolling_std(window_size=long_window).alias('_term_structure_long_vol'),
        ])
        .with_columns(
            (
                (
                    (pl.col('_term_structure_short_vol') / (pl.col('_term_structure_medium_vol') + EPSILON))
                    + (pl.col('_term_structure_medium_vol') / (pl.col('_term_structure_long_vol') + EPSILON))
                ) / 2.0
            ).alias('volatility_term_structure')
        )
        .drop([
            '_term_structure_returns',
            '_term_structure_short_vol',
            '_term_structure_medium_vol',
            '_term_structure_long_vol',
        ])
    )
