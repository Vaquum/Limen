import polars as pl



def exit_quality(data: pl.DataFrame,
                exit_quality_high: float = 1.0,
                exit_quality_low: float = 0.2,
                exit_quality_medium: float = 0.5) -> pl.DataFrame:
    
    '''
    Compute exit quality score based on exit reason and net return.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'exit_reason', 'exit_net_return' columns
        exit_quality_high (float): Quality score for good exits (target hit, profitable trailing stop)
        exit_quality_low (float): Quality score for bad exits (stop loss, unprofitable timeout)
        exit_quality_medium (float): Quality score for neutral exits
        
    Returns:
        pl.DataFrame: The input data with a new column 'exit_quality'
    '''
    
    return data.with_columns([
        pl.when((pl.col('exit_reason').is_in(['target_hit', 'trailing_stop'])) & (pl.col('exit_net_return') > 0))
            .then(pl.lit(exit_quality_high))
            .when((pl.col('exit_reason') == 'stop_loss') | ((pl.col('exit_reason') == 'timeout') & (pl.col('exit_net_return') < 0)))
            .then(pl.lit(exit_quality_low))
            .otherwise(pl.lit(exit_quality_medium))
            .alias('exit_quality')
    ])