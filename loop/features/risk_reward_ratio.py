import polars as pl

# Constants
EPSILON = 0.001  # Small value to prevent division by zero


def risk_reward_ratio(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute risk-reward ratio from capturable breakout and maximum drawdown.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'capturable_breakout', 'max_drawdown' columns
        
    Returns:
        pl.DataFrame: The input data with a new column 'risk_reward_ratio'
    '''
    
    return data.with_columns([
        (pl.col('capturable_breakout') / (pl.col('max_drawdown').abs() + EPSILON))
        .alias('risk_reward_ratio')
    ])