import loop
from loop.sfm.reference import lightgbm
import numpy as np
import polars as pl
import lightgbm as lgb

# Import your quantile model function
from loop.sfm.lightgbm.utils.quantile_model_with_confidence import quantile_model_with_confidence

def test_quantile_model():
    '''
    Quick test for quantile model with confidence filtering
    '''
    
    # Step 1: Get small amount of historical data
    historical = loop.HistoricalData()
    historical.get_historical_klines(
        n_rows=200,  # Small for fast testing
        kline_size=7200,  # 2h klines
        start_date_limit='2024-01-01',
        futures=True
    )
    
    # Step 2: Create simple labels
    df_labeled = create_simple_regression_labels(historical.data)
    
    # Step 3: Convert to Polars if needed
    if not hasattr(df_labeled, 'with_columns'):
        df_labeled = pl.from_pandas(df_labeled)
    
    # Step 4: Run UEL with quantile model
    uel = loop.UniversalExperimentLoop(df_labeled, lightgbm)
    uel.run(
        experiment_name="quantile_model_test",
        n_permutations=3,  # Just 3 for quick test
        random_search=True,
        prep=prep_for_quantile_test,
        model=model_with_quantile
    )
    
    return uel

def create_simple_regression_labels(data):
    '''
    Create simple regression target for testing
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
    
    # Create simple target: future price change percentage
    df = df.with_columns([
        ((pl.col('average_price').shift(-6) - pl.col('average_price')) / pl.col('average_price') * 100).alias('price_change_6h')
    ])
    
    # Fill nulls and drop them
    df = df.with_columns([
        pl.col('price_change_6h').fill_null(0)
    ])
    df = df.drop_nulls()
    
    return df

def prep_for_quantile_test(data, round_params=None):
    '''
    Simple prep function for quantile model testing
    '''
    # Take small sample
    n = len(data)
    sample_size = min(100, n // 2)
    
    if n > sample_size:
        df = data[:sample_size]  # Just take first N rows for simplicity
    else:
        df = data
    
    # Create simple lag features
    for lag in range(1, 4):  # Just 3 lags
        df = df.with_columns([
            pl.col('price_change_6h').shift(lag).alias(f'price_lag_{lag}'),
            pl.col('average_price').shift(lag).alias(f'avg_price_lag_{lag}'),
        ])
    
    # Drop nulls
    df = df.drop_nulls()
    
    # Define feature columns
    feature_cols = [c for c in df.columns if c not in ['datetime', 'price_change_6h']]
    
    # Simple sequential split
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
    
    y_train = train_df['price_change_6h'].to_numpy().astype('float32')
    y_val = val_df['price_change_6h'].to_numpy().astype('float32')
    y_test = test_df['price_change_6h'].to_numpy().astype('float32')
    
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

def model_with_quantile(data, round_params):
    '''
    UEL model function that uses quantile model with confidence
    '''
    # Set some default params if none provided
    if round_params is None:
        round_params = {}
    
    round_params.update({
        'verbose': -1,
        'num_leaves': 15,  # Small for fast testing
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
    })
    
    # Call the quantile model function
    return quantile_model_with_confidence(
        data=data,
        round_params=round_params,
        quantiles=[0.1, 0.5, 0.9],
        confidence_threshold=5.0,  # Stricter for testing
        verbose=True
    )

if __name__ == "__main__":
    try:
        test_quantile_model()
        print("✅ quantile model: ALL TESTS PASSED")
    except Exception as e:
        print(f"❌ quantile model: FAILED - {e}")