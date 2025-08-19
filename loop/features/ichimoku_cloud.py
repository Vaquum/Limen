import polars as pl


def ichimoku_cloud(
        data: pl.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ) -> pl.DataFrame:

    '''
    Compute Ichimoku Cloud components for trend and momentum analysis.

    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        tenkan_period (int): Lookback period for Tenkan-sen
        kijun_period (int): Lookback period for Kijun-sen
        senkou_b_period (int): Lookback period for Senkou Span B
        displacement (int): Number of periods to shift Senkou spans and Chikou span
        
    Returns:
        pl.DataFrame: The input data with new columns:
            'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'
    '''
    
    return (
        data
        .with_columns(
            ((pl.col('high').rolling_max(tenkan_period) + pl.col('low').rolling_min(tenkan_period)) / 2).alias('tenkan')
        )
        .with_columns(
            ((pl.col('high').rolling_max(kijun_period) + pl.col('low').rolling_min(kijun_period)) / 2).alias('kijun')
        )
        .with_columns(
            ((pl.col('tenkan') + pl.col('kijun')) / 2).shift(-displacement).alias('senkou_a')
        )
        .with_columns(
            ((pl.col('high').rolling_max(senkou_b_period) + pl.col('low').rolling_min(senkou_b_period)) / 2).shift(-displacement).alias('senkou_b')
        )
        .with_columns(
            pl.col('close').shift(displacement).alias('chikou')
        )
    )
