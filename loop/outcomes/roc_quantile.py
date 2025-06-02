import polars as pl

from loop.indicators import roc
from loop.indicators import quantile_flag



def roc_quantile(historical, round_params):

    # add outcome
    data = roc(historical.data).filter(~pl.col("roc").is_nan())
    # TODO: Fix the leakage issue
    data = quantile_flag(data=data,
                         col='roc',
                         q=round_params['q'])
    
    data = data.with_columns(
        pl.col("quantile_flag")
          .shift(round_params['shift'])
          .alias("quantile_flag")).drop_nulls("quantile_flag")