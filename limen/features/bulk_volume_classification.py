import polars as pl
from scipy.special import ndtr


def bulk_volume_classification(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Split each bar's volume into buy and sell volume by bulk-volume classification.

    The close-to-close log return is standardized by its rolling standard
    deviation and passed through the standard normal CDF to estimate the buy
    fraction (Easley, Lopez de Prado, and O'Hara). Buy volume is that fraction
    times total volume; sell volume is the remainder. Direction is inferred
    from price action alone, so no trade-side data is required.

    Args:
        data (pl.DataFrame): Klines dataset with close and volume columns
        window (int): Rolling window for the return-volatility standardization
        close_col (str): Column name used for close-to-close log returns
        volume_col (str): Column name for total traded volume

    Returns:
        pl.DataFrame: The input data with new columns 'bvc_buy_volume' and 'bvc_sell_volume'
    '''

    if window <= 0:
        raise ValueError('bulk_volume_classification window must be positive')

    log_return = pl.col(close_col).log() - pl.col(close_col).shift(1).log()
    rolling_std = log_return.rolling_std(window_size=window)
    standardized = (
        pl.when(rolling_std > 0.0)
        .then(log_return / rolling_std)
        .otherwise(None)
        .alias('_bvc_standardized')
    )

    return (
        data
        .with_columns(standardized)
        .with_columns(
            pl.col('_bvc_standardized')
            .map_batches(_normal_cdf, return_dtype=pl.Float64)
            .alias('_bvc_buy_fraction')
        )
        .with_columns([
            (pl.col(volume_col) * pl.col('_bvc_buy_fraction')).alias('bvc_buy_volume'),
            (pl.col(volume_col) * (1.0 - pl.col('_bvc_buy_fraction'))).alias('bvc_sell_volume'),
        ])
        .drop(['_bvc_standardized', '_bvc_buy_fraction'])
    )


def _normal_cdf(series: pl.Series) -> pl.Series:
    return pl.Series(ndtr(series.to_numpy())).fill_nan(None)
