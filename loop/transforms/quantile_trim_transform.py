import polars as pl


def quantile_trim_transform(df: pl.DataFrame, *, time_col: str = 'datetime'):

    '''
    Trim outliers by removing rows outside fixed quantile bounds
    across all numeric columns (AND condition).

    Args:
        df (pl.DataFrame): The input DataFrame.
        time_col (str): The name of the time column to exclude from numeric transforms.

    Returns:
        pl.DataFrame: The DataFrame filtered to rows within bounds for all numeric columns.
    '''

    num_cols = [c for c, dt in zip(df.columns, df.dtypes)
                if dt.is_numeric() and c != time_col]

    if not num_cols:
        return df

    lower_q = 0.005
    upper_q = 0.995

    bounds = df.select([
        pl.col(c).quantile(lower_q).alias(f'{c}__lo') for c in num_cols
    ] + [
        pl.col(c).quantile(upper_q).alias(f'{c}__hi') for c in num_cols
    ])

    lo = {c: float(bounds[0, f'{c}__lo']) for c in num_cols}
    hi = {c: float(bounds[0, f'{c}__hi']) for c in num_cols}

    # Build AND mask across numeric columns
    mask = None
    for c in num_cols:
        cond = pl.col(c).is_between(lo[c], hi[c]) | pl.col(c).is_null()
        mask = cond if mask is None else (mask & cond)

    return df.filter(mask) if mask is not None else df


