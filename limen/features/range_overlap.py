import polars as pl

EPSILON = 1e-10


def range_overlap(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
) -> pl.DataFrame:

    '''
    Compute how much the current bar overlaps the previous bar's range.

    Args:
        data (pl.DataFrame): Klines dataset with high and low price columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices

    Returns:
        pl.DataFrame: The input data with a new column 'range_overlap'
    '''

    overlap = (
        pl.min_horizontal(pl.col(high_col), pl.col(high_col).shift(1))
        - pl.max_horizontal(pl.col(low_col), pl.col(low_col).shift(1))
    ).clip(lower_bound=0.0)

    previous_range = pl.col(high_col).shift(1) - pl.col(low_col).shift(1)

    return data.with_columns((overlap / (previous_range + EPSILON)).alias('range_overlap'))
