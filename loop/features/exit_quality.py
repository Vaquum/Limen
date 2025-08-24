import polars as pl


def exit_quality(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute exit quality score based on exit reason and net return.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'exit_reason', 'exit_net_return' columns
        
    Returns:
        pl.DataFrame: The input data with a new column 'exit_quality'
    '''
    
    return data.with_columns([
        pl.when((pl.col('exit_reason').is_in(['target_hit', 'trailing_stop'])) & (pl.col('exit_net_return') > 0))
            .then(pl.lit(1.0))
            .when((pl.col('exit_reason') == 'stop_loss') | ((pl.col('exit_reason') == 'timeout') & (pl.col('exit_net_return') < 0)))
            .then(pl.lit(0.2))
            .otherwise(pl.lit(0.5))
            .alias('exit_quality')
    ])