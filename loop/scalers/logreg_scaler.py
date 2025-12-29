import polars as pl

SCALING_RULES = {
    'open': 'standard',
    'close': 'standard',
    'high': 'standard',
    'low': 'standard',
    'volume': 'log_standard',
    'maker_ratio': 'none',
    'no_of_trades': 'log_standard',
    'atr': 'standard',
    'ema_breakout': 'standard',
    'imbalance': 'standard',
    'macd': 'standard',
    'ppo': 'standard',
    'roc': 'standard',
    'vwap': 'standard',
    'wilder_rsi': 'divide_100',
    'returns': 'standard',
    'range_pct': 'standard',
    'close_position': 'standard',
    'body_pct': 'standard',
    'gap_high': 'standard',
    'open_liquidity':  'log_standard',
    'high_liquidity':  'log_standard',
    'low_liquidity':   'log_standard',
    'close_liquidity': 'log_standard',
}

class LogRegScaler:

    '''
    LogRegScaler class for scaling and inverse scaling data.
    '''
    
    def __init__(self, x_train: pl.DataFrame):

        '''
        Initialize the LogRegScaler object.

        Args:
            x_train (pl.DataFrame): The training data
        '''
        
        self.means = {}
        self.stds = {}
        
        for col in x_train.columns:
            
            if col not in SCALING_RULES:
                continue
            
            rule = SCALING_RULES[col]
            
            if rule == 'log_standard':
                self.means[col] = x_train.select(pl.col(col).log1p().mean()).item()
                self.stds[col] = x_train.select(pl.col(col).log1p().std(ddof=0)).item()
            
            elif rule == 'standard':
                self.means[col] = x_train[col].mean()
                self.stds[col] = x_train[col].std(ddof=0)
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:

        '''
        Transform the data using the scaling rules.
        
        Args:
            df (pl.DataFrame): The input DataFrame

        Returns:
            pl.DataFrame: The transformed DataFrame
        '''
        
        exprs = []
        
        for col in df.columns:
            
            if col not in SCALING_RULES:
                continue
            
            rule = SCALING_RULES[col]
            
            if rule == 'standard':
                exprs.append(((pl.col(col) - self.means[col]) / self.stds[col]).alias(col))
            
            elif rule == 'log_standard':
                exprs.append(((pl.col(col).log1p() - self.means[col]) / self.stds[col]).alias(col))
            
            elif rule == 'divide_100':
                exprs.append((pl.col(col) / 100).alias(col))
            
            elif rule == 'none':
                exprs.append(pl.col(col).alias(col))
                
        return df.with_columns(exprs)


def inverse_transform(df: pl.DataFrame, scaler: LogRegScaler) -> pl.DataFrame:

    '''
    Inverse transform the data using the scaling rules.

    Args:
        df (pl.DataFrame): The input DataFrame
        scaler (LogRegScaler): The scaler object

    Returns:
        pl.DataFrame: The inverse transformed DataFrame
    '''
    
    exprs = []
    
    for col in df.columns:
        
        if col not in SCALING_RULES:
            continue
            
        rule = SCALING_RULES[col]
        
        if rule == 'standard':
            exprs.append((pl.col(col) * scaler.stds[col] + scaler.means[col]).alias(col))
        
        elif rule == 'log_standard':
            exprs.append(((pl.col(col) * scaler.stds[col] + scaler.means[col]).exp() - 1).alias(col))
        
        elif rule == 'divide_100':
            exprs.append((pl.col(col) * 100).alias(col))
        
        elif rule == 'none':
            exprs.append(pl.col(col).alias(col))
    
    return df.with_columns(exprs)
