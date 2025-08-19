import polars as pl


def hh_hl_structure_regime(df: pl.DataFrame, window: int = 24, score_threshold: int = 4) -> pl.DataFrame:

    '''
    Classify regime by higher-high / higher-low market structure over a rolling window.

    Args:
        df (pl.DataFrame): Input with 'high','low'
        window (int): Rolling window for structure count
        score_threshold (int): Threshold for Up/Down classification

    Returns:
        pl.DataFrame: With 'regime_hh_hl' in {"Up","Flat","Down"}
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


