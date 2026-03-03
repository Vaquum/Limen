import polars as pl


def mfi(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Money Flow Index (MFI).

    Args:
        data (pl.DataFrame): Klines dataset with high/low/close/volume columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        volume_col (str): Column name for volume
        period (int): MFI period

    Returns:
        pl.DataFrame: The input data with a new column 'mfi_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    out_col = f'mfi_{period}'

    positive_flow = (
        pl.when(pl.col('__mfi_delta').is_null())
        .then(None)
        .when(pl.col('__mfi_delta') > 0.0)
        .then(pl.col('__mfi_raw'))
        .otherwise(0.0)
        .alias('__mfi_pos')
    )
    negative_flow = (
        pl.when(pl.col('__mfi_delta').is_null())
        .then(None)
        .when(pl.col('__mfi_delta') < 0.0)
        .then(pl.col('__mfi_raw'))
        .otherwise(0.0)
        .alias('__mfi_neg')
    )

    total_flow = pl.col('__mfi_pos_sum') + pl.col('__mfi_neg_sum')
    mfi_expr = (
        pl.when(total_flow.is_null())
        .then(None)
        .when(total_flow < 1.0)
        .then(0.0)
        .otherwise(100.0 * (pl.col('__mfi_pos_sum') / total_flow))
        .alias(out_col)
    )

    return (
        data
        .with_columns([
            ((pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3.0).alias('__mfi_tp')
        ])
        .with_columns([
            pl.col('__mfi_tp').diff(1).alias('__mfi_delta'),
            (pl.col('__mfi_tp') * pl.col(volume_col).cast(pl.Float64)).alias('__mfi_raw'),
        ])
        .with_columns([positive_flow, negative_flow])
        .with_columns([
            pl.col('__mfi_pos').rolling_sum(window_size=period).alias('__mfi_pos_sum'),
            pl.col('__mfi_neg').rolling_sum(window_size=period).alias('__mfi_neg_sum'),
        ])
        .with_columns([mfi_expr])
        .drop([
            '__mfi_tp',
            '__mfi_delta',
            '__mfi_raw',
            '__mfi_pos',
            '__mfi_neg',
            '__mfi_pos_sum',
            '__mfi_neg_sum',
        ])
    )
