import polars as pl

def perc_based_breakout(data, round_params):

    '''Adds a boolean column based on two parameters:
       
       `breakout_threshold` and `shift` and these must
       both be present in `round_params`. 
    '''
    
    data = data.with_columns([(
        ((pl.col("high") - pl.col("low")) / pl.col("open") * 100) 
            .gt(round_params['breakout_threshold'])).cast(pl.UInt8)
         .shift(round_params['shift'])
         .alias("perc_breakout")
    ]).drop_nulls("perc_breakout")

    return data
