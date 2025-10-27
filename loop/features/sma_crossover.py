import polars as pl


def sma_crossover(
    df: pl.DataFrame, 
    short_window: int = 10, 
    long_window: int = 30
) -> pl.DataFrame:

    '''
    Compute SMA crossover signals.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        short_window
        long_window
        
    Returns:
        pl.DataFrame: The input data with a new column ''
    '''

    df = (
        df
        .with_columns([
            pl.col("close").rolling_mean(short_window).alias(f"sma_short_{short_window}"),
            pl.col("close").rolling_mean(long_window).alias(f"sma_long_{long_window}")
        ])
        .with_columns([
            (
                (pl.col(f"sma_short_{short_window}") > pl.col(f"sma_long_{long_window}")).cast(pl.Int8)
                - (pl.col(f"sma_short_{short_window}") < pl.col(f"sma_long_{long_window}")).cast(pl.Int8)
            ).alias('sma_relation')
        ])
        .with_columns([
            # Detect crossovers using the diff in sma_relation
            (pl.col('sma_relation') - pl.col('sma_relation').shift(1))
            .alias('crossover')
        ])
        .with_columns([
            pl.when(pl.col('crossover') == 2)
              .then(pl.lit(1))
              .when(pl.col('crossover') == -2)
              .then(pl.lit(-1))
              .otherwise(pl.lit(0))
              .alias('signal')
        ])
    )
    
    return df
