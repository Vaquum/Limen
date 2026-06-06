import polars as pl

from limen.features.bulk_volume_classification import bulk_volume_classification


def order_flow_imbalance(
    data: pl.DataFrame,
    window: int = 20,
    classification_window: int = 20,
    close_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute a rolling bulk-volume order-flow imbalance from classified buy and sell volume.

    Bars are split into buy and sell volume by bulk-volume classification, then
    the net signed volume is summed over `window` and divided by the summed total
    volume, giving a directional flow proxy in [-1, 1]. This is a bar-level proxy
    for order-flow imbalance inferred from price; it is not the level-2 order-book
    OFI of Cont, Kukanov, and Stoikov.

    Args:
        data (pl.DataFrame): Klines dataset with close and volume columns
        window (int): Rolling window over which signed flow is accumulated
        classification_window (int): Rolling window for the bulk-volume classification standardization
        close_col (str): Column name used for close-to-close log returns
        volume_col (str): Column name for total traded volume

    Returns:
        pl.DataFrame: The input data with classified buy/sell volume columns and a
            new 'order_flow_imbalance' column
    '''

    if window <= 0:
        raise ValueError('order_flow_imbalance window must be positive')
    if classification_window <= 0:
        raise ValueError('order_flow_imbalance classification_window must be positive')

    classified = bulk_volume_classification(
        data,
        window=classification_window,
        close_col=close_col,
        volume_col=volume_col,
    )

    net_flow = pl.col('bvc_buy_volume') - pl.col('bvc_sell_volume')
    total_flow = pl.col('bvc_buy_volume') + pl.col('bvc_sell_volume')

    rolling_net = net_flow.rolling_sum(window_size=window)
    rolling_total = total_flow.rolling_sum(window_size=window)

    return classified.with_columns(
        pl.when(rolling_total > 0.0)
        .then(rolling_net / rolling_total)
        .otherwise(None)
        .alias('order_flow_imbalance')
    )
