import polars as pl


def hh_hl_structure_regime(df: pl.DataFrame,
                           window: int = 24,
                           score_threshold: int = 4) -> pl.DataFrame:

    '''
    Compute regime by higher-high / higher-low market structure within a rolling window.

    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low' columns
        window (int): Rolling window size for structure count
        score_threshold (int): Absolute score threshold for Up/Down classification

    Returns:
        pl.DataFrame: The input data with a new column 'regime_hh_hl'
    '''

    hh = (pl.col('high') > pl.col('high').shift(1)).cast(pl.Int8)
    hl = (pl.col('low')  > pl.col('low').shift(1)).cast(pl.Int8)
    lh = (pl.col('high') < pl.col('high').shift(1)).cast(pl.Int8)
    ll = (pl.col('low')  < pl.col('low').shift(1)).cast(pl.Int8)

    score = (hh + hl - lh - ll).rolling_sum(window_size=window)

    return df.with_columns([
        pl.when(score >= score_threshold).then(pl.lit('Up'))
         .when(score <= -score_threshold).then(pl.lit('Down'))
         .otherwise(pl.lit('Flat')).alias('regime_hh_hl')
    ])
