import polars as pl


def stochastic_oscillator(
    df: pl.DataFrame,
    window_k: int = 14,
    window_d: int = 3,
) -> pl.DataFrame:

    '''
    Compute
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        window_k
        window_d
        
    Returns:
        pl.DataFrame: The input data with a new column ''
    '''

    highest_col = f"stoch_highest_{window_k}"
    lowest_col = f"stoch_lowest_{window_k}"
    k_col = f"stoch_k_{window_k}"
    d_col = f"stoch_d_{window_d}"

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
    )
