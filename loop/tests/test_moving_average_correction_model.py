import loop
from loop.sfm.reference import lightgbm
import numpy as np
import polars as pl
import lightgbm as lgb

# Import your moving average correction function
from loop.sfm.lightgbm.utils.moving_average_correction_model import moving_average_correction_model

def test_moving_average_correction():
    '''
    Quick test for moving average correction model
    '''
    
    # Step 1: Get small amount of historical data
    historical = loop.HistoricalData()
    historical.get_futures_klines(
        n_rows=300,  # Slightly larger for correction testing
        kline_size=7200,  # 2h klines
        start_date_limit='2024-01-01'
    )
    
    # Step 2: Create labels with some trend/pattern
    df_labeled = create_trending_regression_labels(historical.data)
    
    # Step 3: Convert to Polars if needed
    if not hasattr(df_labeled, 'with_columns'):
        df_labeled = pl.from_pandas(df_labeled)
    
    # Step 4: Run UEL with moving average correction
    uel = loop.UniversalExperimentLoop(df_labeled, lightgbm)
    uel.run(
        experiment_name="ma_correction_test",
        n_permutations=3,  # Just 3 for quick test
        random_search=True,
        prep=prep_for_ma_test,
        model=model_with_ma_correction
    )
    
    return uel

def create_trending_regression_labels(data):
    '''
    Create regression target with some trending patterns for correction testing
    '''
    # Ensure we're working with Polars DataFrame
    if hasattr(data, 'to_pandas'):
        df = data
    else:
        df = pl.from_pandas(data)
    
    # Create average price
    df = df.with_columns([
        ((pl.col('open') + pl.col('high') + pl.col('low') + pl.col('close')) / 4).alias('average_price')
    ])
    
    # Create target with some cyclical pattern that could benefit from MA correction
    df = df.with_columns([
        # Future price change with some trend
        ((pl.col('average_price').shift(-8) - pl.col('average_price')) / pl.col('average_price') * 100).alias('price_change_8h'),
        # Add some cyclical component based on time
        (pl.col('datetime').dt.hour() * 0.1).alias('hour_effect')
    ])
    
    # Combine for a target that has patterns
    df = df.with_columns([
        (pl.col('price_change_8h') + pl.col('hour_effect')).alias('target_with_pattern')
    ])
    
    # Fill nulls and drop them
    df = df.with_columns([
        pl.col('target_with_pattern').fill_null(0)
    ])
    df = df.drop_nulls()
    
    return df

def prep_for_ma_test(data, round_params=None):
    '''
    Prep function for moving average correction testing
    '''
    # Take larger sample for MA correction to work well
    n = len(data)
    sample_size = min(150, n // 2)  # Bigger sample for trend detection
    
    if n > sample_size:
        df = data[:sample_size]  # Sequential for trend testing
    else:
        df = data
    
    # Create features including time-based ones
    df = df.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.weekday().alias('weekday')
    ])
    
    # Create lag features
    for lag in range(1, 5):  # 4 lags
        df = df.with_columns([
            pl.col('target_with_pattern').shift(lag).alias(f'target_lag_{lag}'),
            pl.col('average_price').shift(lag).alias(f'price_lag_{lag}'),
        ])
    
    # Drop nulls
    df = df.drop_nulls()
    
    # Define feature columns
    feature_cols = [c for c in df.columns if c not in ['datetime', 'target_with_pattern', 'price_change_8h', 'hour_effect']]
    
    # Sequential split (important for MA correction testing)
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
    
    y_train = train_df['target_with_pattern'].to_numpy().astype('float32')
    y_val = val_df['target_with_pattern'].to_numpy().astype('float32')
    y_test = test_df['target_with_pattern'].to_numpy().astype('float32')
    
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

def model_with_ma_correction(data, round_params):
    '''
    UEL model function that uses moving average correction
    '''
    # Set some default params if none provided
    if round_params is None:
        round_params = {}
    
    round_params.update({
        'verbose': -1,
        'num_leaves': 20,  # Small for fast testing
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
    })
    
    # Call the moving average correction function
    return moving_average_correction_model(
        data=data,
        round_params=round_params,
        short_window=3,     # Smaller windows for testing
        medium_window=8,
        long_window=15,
        correction_factor=0.3,  # Lighter correction for testing
        trend_threshold=0.1,    # More sensitive trend detection
        residual_window=25,     # Smaller residual window
        verbose=True
    )

if __name__ == "__main__":
    
    try:
        test_moving_average_correction()
        print("✅ moving average correction: ALL TESTS PASSED")
    
    except Exception as e:
        print(f"❌ moving average correction: FAILED - {e}")