from typing import Literal

import polars as pl


ZscoreTransform = Literal['identity', 'log1p', 'abs']


def rolling_zscore(
    data: pl.DataFrame,
    column: str,
    window: int,
    *,
    transform: ZscoreTransform = 'identity',
    output_col: str | None = None,
) -> pl.DataFrame:

    '''
    Compute a rolling z-score for a transformed input column.

    Args:
        data (pl.DataFrame): Dataset containing the source column
        column (str): Source column name
        window (int): Number of periods for the rolling mean and standard deviation
        transform (ZscoreTransform): Optional source transform before standardization
        output_col (str | None): Output column name; defaults to `{column}_zscore_{window}`

    Returns:
        pl.DataFrame: The input data with the rolling z-score column appended
    '''

    if window <= 0:
        raise ValueError('window must be positive')

    x = pl.col(column)
    if transform == 'log1p':
        x = x.log1p()
    elif transform == 'abs':
        x = x.abs()
    elif transform != 'identity':
        raise ValueError("transform must be one of 'identity', 'log1p', or 'abs'")

    mean = x.rolling_mean(window_size=window)
    std = x.rolling_std(window_size=window)
    name = output_col or f'{column}_zscore_{window}'

    return data.with_columns(
        pl.when(std.is_null() | (std == 0.0))
        .then(None)
        .otherwise((x - mean) / std)
        .alias(name)
    )
