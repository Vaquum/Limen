import polars as pl

from limen.features.bulk_volume_classification import bulk_volume_classification


def vpin(
    data: pl.DataFrame,
    window: int = 50,
    classification_window: int = 20,
    close_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute Volume-synchronized Probability of Informed Trading (VPIN).

    Bars are split into buy and sell volume by bulk-volume classification, then
    VPIN is the absolute buy-minus-sell imbalance summed over `window` buckets
    divided by the summed total volume (Easley, Lopez de Prado, and O'Hara). High
    values flag one-sided, toxic-looking flow. For canonical equal-volume buckets,
    supply volume bars; on time bars this is the standard bar-level approximation.

    Args:
        data (pl.DataFrame): Klines dataset with close and volume columns
        window (int): Number of buckets over which the imbalance is accumulated
        classification_window (int): Rolling window for the bulk-volume classification standardization
        close_col (str): Column name used for close-to-close log returns
        volume_col (str): Column name for total traded volume

    Returns:
        pl.DataFrame: The input data with classified buy/sell volume columns and a new 'vpin' column
    '''

    classified = bulk_volume_classification(
        data,
        window=classification_window,
        close_col=close_col,
        volume_col=volume_col,
    )

    order_imbalance = (pl.col('bvc_buy_volume') - pl.col('bvc_sell_volume')).abs()
    total_volume = pl.col('bvc_buy_volume') + pl.col('bvc_sell_volume')

    return classified.with_columns(
        (
            order_imbalance.rolling_sum(window_size=window)
            / total_volume.rolling_sum(window_size=window)
        ).alias('vpin')
    )
