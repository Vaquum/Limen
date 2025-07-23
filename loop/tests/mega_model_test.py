import loop
from loop.sfm import lightgbm
import uuid
import numpy as np
import polars as pl
import lightgbm as lgb
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import your mega model class from the correct path
from loop.sfm.lightgbm.utils.mega_model_data_sampler import run_mega_model_experiment

def test_mega_model_with_live_labeling():
    """
    Custom test that combines:
    1. Live data fetching + labeling (from SFM approach)
    2. MegaModelDataSampler testing
    3. Reduced data size for fast testing
    """

    
    # Step 1: Get historical data (reduced size for testing)
    historical = loop.HistoricalData()
    historical.get_historical_klines(
        n_rows=200,  # Very small for fast testing
        kline_size=7200,  # 2h klines like in SFM
        start_date_limit='2024-01-01',
        futures=True
    )
    
    # Step 2: Convert to average price klines and add labeling
    df_labeled = create_breakout_labels(historical.data)
    
    # Step 3: CONVERT TO POLARS - This is the key fix!
    if not hasattr(df_labeled, 'with_columns'):
        # It's pandas, convert to Polars
        df_labeled = pl.from_pandas(df_labeled)
    
    # Step 4: Run mega model experiment with reduced sample size

    results = run_mega_model_experiment(
        df_orig=df_labeled,
        prep_func=prep_for_mega_model,
        model_func=model_for_mega_model,
        target='breakout_long',
        enable_mega_models=True,
        mega_model_size=2,
        random_state=42,
        sample_size=100,
        n_samples=3  
    )    
    
    return results

def create_breakout_labels(data):
    """
    Create breakout labels similar to SFM approach but simplified for testing
    """
    # Ensure we're working with Polars DataFrame
    if hasattr(data, 'to_pandas'):
        # It's already a Polars DataFrame
        df = data
    else:
        # Convert from pandas if needed
        df = pl.from_pandas(data)
    
    # Convert to average price format (simplified version of to_average_price_klines)
    df = df.with_columns([
        ((pl.col('open') + pl.col('high') + pl.col('low') + pl.col('close')) / 4).alias('average_price')
    ])
    
    # Add simple breakout detection (simplified version of build_breakout_flags)
    # Look ahead 12 periods (24 hours with 2h klines) for breakouts
    lookback = 12
    breakout_threshold = 0.05  # 5% breakout threshold
    
    df = df.with_columns([
        # Calculate future max/min for breakout detection
        pl.col('average_price').shift(-lookback).rolling_max(window_size=lookback).alias('future_max'),
        pl.col('average_price').shift(-lookback).rolling_min(window_size=lookback).alias('future_min'),
    ])
    
    # Calculate breakout percentages
    df = df.with_columns([
        ((pl.col('future_max') - pl.col('average_price')) / pl.col('average_price')).alias('breakout_long'),
        ((pl.col('average_price') - pl.col('future_min') ) / pl.col('average_price')).alias('breakout_short'),
    ])
    
    # Fill nulls with 0 and ensure non-negative
    df = df.with_columns([
        pl.col('breakout_long').fill_null(0).clip(lower_bound=0),
        pl.col('breakout_short').fill_null(0).clip(lower_bound=0),
    ])
    
    # Drop rows with nulls
    df = df.drop_nulls()
    
    return df

def prep_for_mega_model(data, round_params=None):
    """
    UEL-compatible prep function - must accept data and round_params
    """
    # Take a random window for testing (similar to SFM approach)
    n = len(data)
    sample_size = min(100, n // 2)  # Very small for testing - 100 or half the data
    
    if n > sample_size:
        # Random contiguous window
        lo = int(n * 0.05)
        hi = int(n * 0.95) - sample_size
        if hi >= lo:
            rng = np.random.default_rng(42)  # Fixed seed for testing
            start = int(rng.integers(lo, hi + 1))
            df = data[start:start + sample_size]
        else:
            df = data
    else:
        df = data
    
    # Create simple lag features
    lookback = 3  # Much smaller for testing
    
    for lag in range(1, lookback + 1):
        df = df.with_columns([
            pl.col('breakout_long').shift(lag).alias(f'breakout_long_lag_{lag}'),
            pl.col('breakout_short').shift(lag).alias(f'breakout_short_lag_{lag}'),
            pl.col('average_price').shift(lag).alias(f'price_lag_{lag}'),
        ])
    
    # Add rolling features
    df = df.with_columns([
        pl.col('breakout_long').shift(1).rolling_mean(window_size=lookback).alias('breakout_long_mean'),
        pl.col('breakout_short').shift(1).rolling_mean(window_size=lookback).alias('breakout_short_mean'),
    ])
    
    # Drop nulls
    df = df.drop_nulls()
    
    # Define feature columns (excluding target and datetime)
    feature_cols = [c for c in df.columns if c not in ['datetime', 'breakout_long', 'breakout_short'] and not c.startswith('future_')]
    
    # Sequential split
    n_total = len(df)
    train_end = int(n_total * 0.6)
    val_end = int(n_total * 0.8)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Convert to numpy
    x_train = train_df.select(feature_cols).to_numpy().astype('float32')
    x_val = val_df.select(feature_cols).to_numpy().astype('float32')
    x_test = test_df.select(feature_cols).to_numpy().astype('float32')
    
    y_train = train_df['breakout_long'].to_numpy().astype('float32')
    y_val = val_df['breakout_long'].to_numpy().astype('float32')
    y_test = test_df['breakout_long'].to_numpy().astype('float32')
    
    # Create LightGBM datasets
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
    
    return {
        'dtrain': dtrain,
        'dval': dval,
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }

def model_for_mega_model(data, round_params):
    """
    UEL-compatible model function - must accept data and round_params
    """
    # Set default params if none provided
    if round_params is None:
        round_params = {}
    
    round_params.update({
        'objective': 'regression',
        'metric': 'mae',
        'verbose': -1,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    })
    
    model = lgb.train(
        params=round_params,
        train_set=data['dtrain'],
        num_boost_round=20,  # Much smaller for testing
        valid_sets=[data['dval']],
        callbacks=[lgb.early_stopping(5, verbose=False)]  # Early stopping after 5 rounds
    )
    
    # Make predictions
    y_pred = model.predict(data['x_test'])
    
    # Calculate metrics
    mae = mean_absolute_error(data['y_test'], y_pred)
    rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))
    r2 = r2_score(data['y_test'], y_pred)
    
    # Return in UEL expected format
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'models': [model],
        'extras': {'rmse': rmse, 'mae': mae, 'r2': r2}
    }