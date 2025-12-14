#!/usr/bin/env python3
"""
Parameter sweep for baseline breakout model (no directional conditional)
Find best return on 3-month test period (2024-Q1)
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')
sys.path.insert(0, '/Users/beyondsyntax/Loop')

import loop
import polars as pl
from datetime import datetime, timedelta, timezone
from loop.sfm.lightgbm.utils.tradeline_long_binary import apply_long_only_exit_strategy, calculate_atr
import numpy as np
import lightgbm as lgb
from itertools import product

print('=' * 80)
print('BASELINE PARAMETER SWEEP FOR BEST RETURN')
print('=' * 80)

# Load data for 2024-Q1 test period (need 17 months before for training)
print('\nLoading data...')
test_start = datetime(2024, 1, 1)
test_end = datetime(2024, 3, 31)
train_start = test_start - timedelta(days=17 * 30)

print(f'Train start: {train_start.strftime("%Y-%m-%d")}')
print(f'Test period: {test_start.strftime("%Y-%m-%d")} to {test_end.strftime("%Y-%m-%d")}')

# Load data month by month
all_data_frames = []
current_date = train_start

while current_date <= datetime.now():
    if current_date > test_end:
        break

    month_str = current_date.strftime('%Y-%m')
    print(f'  Loading {month_str}...', end=' ', flush=True)

    historical = loop.HistoricalData()
    historical.get_spot_klines(
        kline_size=300,
        start_date_limit=current_date.strftime('%Y-%m-%d'),
        n_rows=10000
    )

    if len(historical.data) > 0:
        if current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1, tzinfo=timezone.utc)

        month_data = historical.data.filter(
            (pl.col('datetime') >= pl.lit(current_date.replace(tzinfo=timezone.utc))) &
            (pl.col('datetime') < pl.lit(next_month))
        )

        if len(month_data) > 0:
            all_data_frames.append(month_data)
            print(f'{len(month_data):,} candles')
        else:
            print('0 candles')
    else:
        print('0 candles')

    # Move to next month
    if current_date.month == 12:
        current_date = datetime(current_date.year + 1, 1, 1)
    else:
        current_date = datetime(current_date.year, current_date.month + 1, 1)

print('\nCombining data...')
full_data = pl.concat(all_data_frames)
print(f'Total: {len(full_data):,} candles')
print(f'Range: {full_data["datetime"].min()} to {full_data["datetime"].max()}')

# Define parameter grid (reduced to ~192 combinations)
param_grid = {
    'threshold_pct': [0.005, 0.01],
    'lookahead_hours': [72],  # Use best from directional
    'quantile_threshold': [0.7, 0.75, 0.8],
    'min_height_pct': [0.003, 0.005],
    'max_duration_hours': [48, 72],
    'n_estimators': [200, 300],
    'num_leaves': [31, 63],
    'learning_rate': [0.05, 0.1],
}

# Generate all combinations
param_names = list(param_grid.keys())
param_values = [param_grid[name] for name in param_names]
all_combinations = list(product(*param_values))

print(f'\nTotal parameter combinations: {len(all_combinations)}')
print(f'Testing on 2024-Q1 (3 months)')

# Function to create features for baseline model
def create_baseline_features(df, threshold_pct, lookahead_hours, quantile_threshold,
                             min_height_pct, max_duration_hours):
    """Create features and labels for baseline breakout detection"""
    df = df.sort('datetime')

    # Calculate returns
    df = df.with_columns([
        (pl.col('close').shift(-1) / pl.col('close') - 1).alias('return_1h'),
        (pl.col('close').shift(-6) / pl.col('close') - 1).alias('return_6h'),
        (pl.col('close').shift(-12) / pl.col('close') - 1).alias('return_12h'),
        (pl.col('close').shift(-24) / pl.col('close') - 1).alias('return_24h'),
    ])

    # Calculate max return in lookahead window
    n_candles = int(lookahead_hours / 5)  # 5-min candles
    future_returns = []
    for i in range(1, n_candles + 1):
        future_returns.append(pl.col('close').shift(-i) / pl.col('close') - 1)

    df = df.with_columns([
        pl.max_horizontal(future_returns).alias('max_future_return')
    ])

    # Create label: breakout = max_future_return > threshold_pct
    df = df.with_columns([
        (pl.col('max_future_return') > threshold_pct).cast(pl.Int32).alias('label')
    ])

    # Technical features
    df = df.with_columns([
        # Price features
        (pl.col('close') / pl.col('open') - 1).alias('candle_return'),
        (pl.col('high') - pl.col('low')) / (pl.col('close') + 1e-10).alias('range_pct'),
        ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).clip(0, 1).alias('close_position'),

        # Volume
        (pl.col('volume') / pl.col('volume').rolling_mean(24)).alias('volume_ratio_24h'),

        # Moving averages
        (pl.col('close') / pl.col('close').rolling_mean(6) - 1).alias('ma_6_dist'),
        (pl.col('close') / pl.col('close').rolling_mean(12) - 1).alias('ma_12_dist'),
        (pl.col('close') / pl.col('close').rolling_mean(24) - 1).alias('ma_24_dist'),

        # Momentum
        (pl.col('close') / pl.col('close').shift(6) - 1).alias('momentum_6h'),
        (pl.col('close') / pl.col('close').shift(12) - 1).alias('momentum_12h'),
        (pl.col('close') / pl.col('close').shift(24) - 1).alias('momentum_24h'),
    ])

    # Drop rows with nulls
    df = df.drop_nulls()

    return df

# Trading config
exit_config = {
    'confidence_threshold': 0.5,
    'position_size': 0.1,
    'min_stop_loss': 0.01,
    'max_stop_loss': 0.05,
    'atr_stop_multiplier': 2.0,
    'trailing_activation': 0.01,
    'trailing_distance': 0.5,
    'loser_timeout_hours': 12,
    'max_hold_hours': 96,
    'default_atr_pct': 0.01,
    'initial_capital': 100000.0
}

results_list = []

# Run sweep
for i, combo in enumerate(all_combinations):
    params = dict(zip(param_names, combo))

    if (i + 1) % 10 == 0:
        print(f'\nTesting permutation {i + 1}/{len(all_combinations)}...')

    try:
        # Create features
        df_features = create_baseline_features(
            full_data.clone(),
            params['threshold_pct'],
            params['lookahead_hours'],
            params['quantile_threshold'],
            params['min_height_pct'],
            params['max_duration_hours']
        )

        # Split data: 70% train, 15% val, 15% test
        n_total = len(df_features)
        train_end_idx = int(n_total * 0.7)
        val_end_idx = int(n_total * 0.85)

        # Feature columns (removed range_pct due to inf/nan issues)
        feature_cols = [
            'candle_return', 'close_position',
            'volume_ratio_24h', 'ma_6_dist', 'ma_12_dist', 'ma_24_dist',
            'momentum_6h', 'momentum_12h', 'momentum_24h'
        ]

        # Train data
        train_data = df_features[:train_end_idx]
        X_train = train_data.select(feature_cols).to_numpy()
        y_train = train_data['label'].to_numpy()

        # Val data
        val_data = df_features[train_end_idx:val_end_idx]
        X_val = val_data.select(feature_cols).to_numpy()
        y_val = val_data['label'].to_numpy()

        # Test data
        test_data = df_features[val_end_idx:]
        X_test = test_data.select(feature_cols).to_numpy()
        y_test = test_data['label'].to_numpy()

        # Train LightGBM model
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': params['n_estimators'],
            'num_leaves': params['num_leaves'],
            'learning_rate': params['learning_rate'],
            'max_depth': -1,
            'min_child_samples': 20,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # Get predictions on test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Prepare test data for trading simulation
        test_ohlcv = test_data.select(['datetime', 'open', 'high', 'low', 'close', 'volume', 'label'])
        test_ohlcv = calculate_atr(test_ohlcv, period=24)

        # Run trading simulation
        _, trading_results = apply_long_only_exit_strategy(
            test_ohlcv, y_pred, y_pred_proba, params['threshold_pct'], exit_config
        )

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5

        # Store results
        result = {
            **params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'return_pct': trading_results['total_return_net_pct'],
            'win_rate': trading_results['trade_win_rate_pct'],
            'trades': trading_results['trades_count'],
            'avg_win': trading_results['avg_win'],
            'avg_loss': trading_results['avg_loss']
        }
        results_list.append(result)

    except Exception as e:
        print(f'  Error on permutation {i + 1}: {str(e)}')
        continue

print('\n' + '=' * 80)
print('SWEEP COMPLETE')
print('=' * 80)

if results_list:
    # Create results dataframe
    results_df = pl.DataFrame(results_list)

    # Sort by return
    results_df = results_df.sort('return_pct', descending=True)

    # Save full results
    output_file = '/Users/beyondsyntax/Loop/catalysis/baseline_param_sweep_results.csv'
    results_df.write_csv(output_file)
    print(f'\nFull results saved to: {output_file}')

    # Display top 10 by return
    print('\n' + '=' * 80)
    print('TOP 10 BY RETURN:')
    print('=' * 80)
    print(results_df.head(10))

    # Display best result
    print('\n' + '=' * 80)
    print('BEST PARAMETERS BY RETURN:')
    print('=' * 80)
    best = results_df.row(0, named=True)
    for k, v in best.items():
        print(f'  {k:25s}: {v}')

    # Compare with directional best (0.619 AUC, 5.84% return on 2024-Q1)
    print('\n' + '=' * 80)
    print('COMPARISON WITH DIRECTIONAL MODEL:')
    print('=' * 80)
    print(f'Directional (2024-Q1): AUC 0.6730, Return 5.84%, Win 64.93%, 653 trades')
    print(f'Baseline Best:         AUC {best["auc"]:.4f}, Return {best["return_pct"]:.2f}%, Win {best["win_rate"]:.2f}%, {best["trades"]:.0f} trades')

else:
    print('\nNo successful results')

print('\n' + '=' * 80)
print('Done!')
