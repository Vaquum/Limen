import polars as pl


def gap_high(data):

    '''Compute the gap between the high and close prices.
    
    Args:
        data (pl.DataFrame): The input data.

    Returns:
        pl.DataFrame: The input data with the gap between the high and close prices.
    '''

    return data.with_columns(((pl.col('high') - pl.col('close').shift(1)) / pl.col('close').shift(1)).alias('gap_high'))


def returns(data):

    '''Compute the returns of the close prices.
    
    Args:
        data (pl.DataFrame): The input data.

    Returns:
        pl.DataFrame: The input data with the returns of the close prices.
    '''

    return data.with_columns(pl.col('close').pct_change().alias('returns'))


def close_position(data):

    '''Compute the close position of the close prices.
    
    Args:
        data (pl.DataFrame): The input data.

    Returns:
        pl.DataFrame: The input data with the close position of the close prices.
    '''

    return data.with_columns(((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-8)).alias('close_position'))


def body_pct(data):

    '''Calculate the body percentage of the close prices.
    
    Args:
        data (pl.DataFrame): The input data.

    Returns:
        pl.DataFrame: The input data with the body percentage of the close prices.
    '''

    return data.with_columns(((pl.col('close') - pl.col('open')) / pl.col('open')).alias('body_pct'))


def range_pct(data):

    '''Calculate the range percentage of the close prices.
    
    Args:
        data (pl.DataFrame): The input data.

    Returns:
        pl.DataFrame: The input data with the range percentage of the close prices.
    '''

    return data.with_columns(((pl.col('high') - pl.col('low')) / pl.col('close')).alias('range_pct'))
