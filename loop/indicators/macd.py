import polars as pl


def macd(data: pl.DataFrame,
         close_col: str = 'close',
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9,
         name: str = 'macd',
         signal_name: str = 'macd_signal',
         hist_name: str = 'macd_hist') -> pl.DataFrame:
    
    '''
    Compute MACD (Moving Average Convergence Divergence) indicator.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        close_col (str): Column name for close prices
        fast_period (int): Period for fast EMA calculation
        slow_period (int): Period for slow EMA calculation
        signal_period (int): Period for signal line EMA calculation
        name (str): Alias name for the MACD output column
        signal_name (str): Alias name for the MACD signal output column
        hist_name (str): Alias name for the MACD histogram output column

    Returns:
        pl.DataFrame: The input data with three columns: '{name}', '{signal_name}', '{hist_name}'
    '''

    alpha_fast = 2.0 / (fast_period + 1)
    alpha_slow = 2.0 / (slow_period + 1)
    alpha_signal = 2.0 / (signal_period + 1)

    return (
        data
        .with_columns([
            pl.col(close_col)
              .ewm_mean(alpha=alpha_fast, adjust=False)
              .alias('__ema_fast'),
            pl.col(close_col)
              .ewm_mean(alpha=alpha_slow, adjust=False)
              .alias('__ema_slow')
        ])
        .with_columns([
            (pl.col('__ema_fast') - pl.col('__ema_slow'))
              .alias(name)
        ])
        .with_columns([
            pl.col(name)
              .ewm_mean(alpha=alpha_signal, adjust=False)
              .alias(signal_name)
        ])
        .with_columns([
            (pl.col(name) - pl.col(signal_name))
              .alias(hist_name)
        ])
        .drop(['__ema_fast', '__ema_slow'])
    )
