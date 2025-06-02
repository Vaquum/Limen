import polars as pl


def quantiles(data: pl.DataFrame, column: str):

    '''
    Get the quantiles of column values in a dataframe.

    Args:
        data (pl.DataFrame): The dataframe to get the quantiles of.
        column (str): The column to get the quantiles of.

    Returns:
        pl.DataFrame: A dataframe with the quantiles of the column.

    '''

    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999]
    
    out = []
    
    for quantile in quantiles:
        out.append([quantile, data.select(pl.col(column).quantile(quantile)).item()])

    df_quantiles = pl.DataFrame(out, orient='row')
    df_quantiles.columns = ['quantile', column]

    return df_quantiles
