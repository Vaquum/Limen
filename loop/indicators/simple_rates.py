import polars as pl

def gap_high(data):
    return data.with_columns(((pl.col('high') - pl.col('close').shift(1)) / pl.col('close').shift(1)).alias('gap_high'))

def returns(data):
    return data.with_columns(pl.col('close').pct_change().alias('returns'))

def close_position(data):
    return data.with_columns(((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-8)).alias('close_position'))

def body_pct(data):
    return data.with_columns(((pl.col('close') - pl.col('open')) / pl.col('open')).alias('body_pct'))

def range_pct(data):
    return data.with_columns(((pl.col('high') - pl.col('low')) / pl.col('close')).alias('range_pct'))
