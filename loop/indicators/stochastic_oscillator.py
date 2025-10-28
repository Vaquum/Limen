import polars as pl


def stochastic_oscillator(
    df: pl.DataFrame,
    window_k: int = 14,
    window_d: int = 3,
) -> pl.DataFrame:
    
    '''
    Compute Stochastic Oscillator (%K and %D) using rolling highs and lows.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        window_k (int): Number of periods for %K calculation
        window_d (int): Number of periods for %D smoothing (SMA of %K)
        
    Returns:
        pl.DataFrame: The input data with two new columns 'stoch_k' and 'stoch_d'
    '''
    
    highest_col = 'stoch_highest'
    lowest_col = 'stoch_lowest'
    k_col = 'stoch_k'
    d_col = 'stoch_d'

    return (
        df.with_columns([
            pl.col('high').rolling_max(window_k).alias(highest_col),
            pl.col('low').rolling_min(window_k).alias(lowest_col)
        ])
        .with_columns(
            (
                (pl.col('close') - pl.col(lowest_col))
                / (pl.col(highest_col) - pl.col(lowest_col))
                * 100
            ).alias(k_col)
        )
        .with_columns(
            pl.col(k_col).rolling_mean(window_d).alias(d_col)
        )
        .drop([highest_col, lowest_col])
    )
