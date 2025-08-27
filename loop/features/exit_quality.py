import polars as pl

# Exit quality score constants
QUALITY_HIGH = 1.0    # Perfect exits: target hit or trailing stop with profit
QUALITY_LOW = 0.2     # Poor exits: stop loss or unprofitable timeout
QUALITY_MEDIUM = 0.5  # Neutral exits: other scenarios


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
            .then(pl.lit(QUALITY_HIGH))
            .when((pl.col('exit_reason') == 'stop_loss') | ((pl.col('exit_reason') == 'timeout') & (pl.col('exit_net_return') < 0)))
            .then(pl.lit(QUALITY_LOW))
            .otherwise(pl.lit(QUALITY_MEDIUM))
            .alias('exit_quality')
    ])