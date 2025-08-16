import polars as pl


def price_change_pct(data: pl.DataFrame, 
                     period: int = 1) -> pl.DataFrame:
    
    '''
    Compute price change percentage over a specific period.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Number of periods to look back

    Returns:
        pl.DataFrame: The input data with a new column 'price_change_pct_{period}'
    '''

    return data.with_columns([
        (
            (pl.col('close') - pl.col('close').shift(period)) /
            pl.col('close').shift(period) * 100
        ).alias(f"price_change_pct_{period}")
    ])
