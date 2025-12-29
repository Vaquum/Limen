import polars as pl


def quantiles(data: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Compute quantile distribution for specified column in dataframe.

    Args:
        data (pl.DataFrame): Polars dataframe containing the column for quantile analysis
        column (str): Column name to compute quantiles for

    Returns:
        pl.DataFrame: Dataframe with quantile values and corresponding column values
    """

    quantiles = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.99,
        0.999,
        0.9999,
        0.99999,
        0.999999,
        0.9999999,
        0.99999999,
    ]

    out = []

    for quantile in quantiles:
        out.append([quantile, data.select(pl.col(column).quantile(quantile)).item()])

    df_quantiles = pl.DataFrame(out, orient="row")
    df_quantiles.columns = ["quantile", column]

    return df_quantiles
