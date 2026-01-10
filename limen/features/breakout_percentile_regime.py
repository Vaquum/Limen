import polars as pl

from limen.features.price_range_position import price_range_position


def breakout_percentile_regime(df: pl.DataFrame,
                               period: int = 24,
                               p_hi: float = 0.85,
                               p_lo: float = 0.15) -> pl.DataFrame:

    '''
    Compute regime classification by percentile position of 'close' within rolling [low, high].

    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        period (int): Rolling window for high/low range
        p_hi (float): Upper percentile threshold in [0, 1]
        p_lo (float): Lower percentile threshold in [0, 1]

    Returns:
        pl.DataFrame: The input data with a new column 'regime_breakout_pct'
    '''

    pos_df = price_range_position(df, period)
    pos = pl.col('price_range_position')
    
    return pos_df.with_columns([
        pl.when(pos >= p_hi).then(pl.lit('Up'))
         .when(pos <= p_lo).then(pl.lit('Down'))
         .otherwise(pl.lit('Flat')).alias('regime_breakout_pct')
    ])
