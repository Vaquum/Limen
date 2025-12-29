import polars as pl


def sma_crossover(
    df: pl.DataFrame, 
    short_window: int = 10, 
    long_window: int = 30,
    crossover_bull: int = 2,
    crossover_bear: int = -2
) -> pl.DataFrame:
    
    '''
    Compute Simple Moving Average (SMA) crossover signals.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        short_window (int): Number of periods for short-term SMA
        long_window (int): Number of periods for long-term SMA
        crossover_bull (int): Value indicating bullish crossover
        crossover_bear (int): Value indicating bearish crossover
        
    Returns:
        pl.DataFrame: The input data with new columns 'crossover', and 'signal'
    '''
    
    return (
        df.with_columns([
            pl.col('close').rolling_mean(short_window).alias('sma_short'),
            pl.col('close').rolling_mean(long_window).alias('sma_long')
        ])
        .with_columns([
            (
                (pl.col('sma_short') > pl.col('sma_long')).cast(pl.Int8)
                - (pl.col('sma_short') < pl.col('sma_long')).cast(pl.Int8)
            ).alias('sma_relation')
        ])
        .with_columns([
            (pl.col('sma_relation') - pl.col('sma_relation').shift(1)).alias('crossover')
        ])
        .with_columns([
            pl.when(pl.col('crossover') == crossover_bull)
              .then(pl.lit(1))
              .when(pl.col('crossover') == crossover_bear)
              .then(pl.lit(-1))
              .otherwise(pl.lit(0))
              .alias('signal')
        ])
        .drop(['sma_short', 'sma_long', 'sma_relation'])
    )
