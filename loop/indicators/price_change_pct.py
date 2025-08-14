import polars as pl


def price_change_pct(data: pl.DataFrame, 
                     period: int = 1,
                     name: str = 'price_change_pct') -> pl.DataFrame:
    
    '''
    Compute price change percentage over a specific period.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Number of periods to look back
        name (str): Alias name for the price change percentage output column

    Returns:
        pl.DataFrame: The input data with a new column '{name}'
    '''

    return data.with_columns([
        (
            (pl.col('close') - pl.col('close').shift(period)) /
            pl.col('close').shift(period) * 100
        ).alias(name)
    ])
