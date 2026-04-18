import polars as pl

EPSILON = 1e-10


def illiquidity_shock(
    data: pl.DataFrame,
    window: int = 20,
    price_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute the current Amihud illiquidity level relative to its trailing baseline.

    Args:
        data (pl.DataFrame): Klines dataset with price and volume columns
        window (int): Number of periods used for the trailing Amihud baseline
        price_col (str): Column name used for close-to-close returns
        volume_col (str): Column name for traded volume

    Returns:
        pl.DataFrame: The input data with a new column 'illiquidity_shock'
    '''

    amihud_expr = pl.col(price_col).pct_change().abs() / ((pl.col(price_col) * pl.col(volume_col)) + EPSILON)

    return (
        data
        .with_columns(amihud_expr.alias('_illiquidity_level'))
        .with_columns(
            (
                pl.col('_illiquidity_level')
                / (pl.col('_illiquidity_level').rolling_mean(window_size=window).shift(1) + EPSILON)
            ).alias('illiquidity_shock')
        )
        .drop('_illiquidity_level')
    )
