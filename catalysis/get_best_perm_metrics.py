#!/usr/bin/env python3
"""
Get all available metrics for best permutation
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')
sys.path.insert(0, '/Users/beyondsyntax/Loop')

import loop
import polars as pl
from datetime import datetime, timedelta
import directional_conditional as dc
from loop.sfm.lightgbm.utils.tradeline_long_binary import apply_long_only_exit_strategy, calculate_atr
import numpy as np

print('=' * 80)
print('GETTING ALL METRICS FOR BEST PERMUTATION')
print('=' * 80)

# Load data
print('\nLoading data...')
kline_size = 300
end_date = datetime.now()
n_months = 20
start_date = end_date - timedelta(days=n_months * 30)
start_date_str = start_date.strftime('%Y-%m-%d')

historical = loop.HistoricalData()
historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
full_data = historical.data

print(f'Loaded {len(full_data):,} candles')

# Best permutation parameters
best_params = {
    'threshold_pct': 0.005,
    'lookahead_hours': 72,
    'quantile_threshold': 0.7,
    'min_height_pct': 0.005,
    'max_duration_hours': 48,
    'conditional_threshold': 0.6,
    'movement_threshold': 0.3,
    'use_safer': True,
    'n_estimators': 300,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0,
    'lambda_l2': 0
}

# Add trading parameters from baseline
trading_params = {
    'confidence_threshold': 0.5,
    'position_size': 0.1,
    'min_stop_loss': 0.01,
    'max_stop_loss': 0.05,
    'atr_stop_multiplier': 2.0,
    'trailing_activation': 0.01,
    'trailing_distance': 0.5,
    'loser_timeout_hours': 12,
    'max_hold_hours': 96,
    'default_atr_pct': 0.01
}

# Merge params
all_params = {**best_params, **trading_params}

# Create custom params function
def custom_params():
    return {k: [v] for k, v in all_params.items()}

# Temporarily replace params
original_params = dc.params
dc.params = custom_params

print('\n' + '=' * 80)
print('RUNNING DIRECTIONAL MODEL')
print('=' * 80)

try:
    # Prep data
    print('\nPreparing data...')
    data_dict = dc.prep(full_data, round_params=best_params)

    # Add ATR and original df for trading
    print('Adding OHLCV data for trading simulation...')

    # Get the cleaned data with OHLC
    df_clean = full_data.clone()

    # Calculate ATR
    df_clean = calculate_atr(df_clean, period=24)

    # Create label placeholder (will be overwritten by predictions)
    df_clean = df_clean.with_columns(pl.lit(0).alias('label'))

    # Get test split indices from data_dict alignment
    test_alignment_indices = data_dict['_alignment']['test']

    # Use the same test indices to get OHLCV data
    test_df = df_clean[test_alignment_indices]
    test_df = test_df.select(['datetime', 'open', 'high', 'low', 'close', 'volume', 'atr_pct', 'label'])

    data_dict['_original_df'] = test_df

    # Run model
    print('Training models and making predictions...')
    results = dc.model(data_dict, round_params=all_params)

    print('\n' + '=' * 80)
    print('RUNNING TRADING SIMULATION')
    print('=' * 80)

    # Get predictions and probabilities
    y_pred = results['_preds']

    # Get signal probability from extras
    probabilities = results['extras']['probabilities']
    if all_params['use_safer']:
        y_proba = probabilities['safe_long_given_movement']
    else:
        y_proba = probabilities['long_given_movement']

    # Create exit config
    exit_config = {
        'confidence_threshold': trading_params['confidence_threshold'],
        'position_size': trading_params['position_size'],
        'min_stop_loss': trading_params['min_stop_loss'],
        'max_stop_loss': trading_params['max_stop_loss'],
        'atr_stop_multiplier': trading_params['atr_stop_multiplier'],
        'trailing_activation': trading_params['trailing_activation'],
        'trailing_distance': trading_params['trailing_distance'],
        'loser_timeout_hours': trading_params['loser_timeout_hours'],
        'max_hold_hours': trading_params['max_hold_hours'],
        'default_atr_pct': trading_params['default_atr_pct'],
        'initial_capital': 100000.0
    }

    # Run trading simulation
    _, trading_results = apply_long_only_exit_strategy(
        test_df, y_pred, y_proba, best_params['threshold_pct'], exit_config
    )

    # Add trading metrics to results
    results['trading_return_net_pct'] = float(trading_results['total_return_net_pct'])
    results['trading_win_rate_pct'] = float(trading_results['trade_win_rate_pct'])
    results['trading_trades_count'] = float(trading_results['trades_count'])
    results['trading_avg_win'] = float(trading_results['avg_win'])
    results['trading_avg_loss'] = float(trading_results['avg_loss'])

    print('\n' + '=' * 80)
    print('ALL METRICS:')
    print('=' * 80)

    # Display classification metrics
    print('\nClassification Metrics:')
    print(f'  recall:    {results["recall"]:.4f}')
    print(f'  precision: {results["precision"]:.4f}')
    print(f'  fpr:       {results["fpr"]:.4f}')
    print(f'  auc:       {results["auc"]:.4f}')
    print(f'  accuracy:  {results["accuracy"]:.4f}')

    # Display trading metrics
    print('\nTrading Metrics:')
    print(f'  return_net_pct:  {results["trading_return_net_pct"]:.2f}%')
    print(f'  win_rate_pct:    {results["trading_win_rate_pct"]:.2f}%')
    print(f'  trades_count:    {results["trading_trades_count"]:.0f}')
    print(f'  avg_win:         {results["trading_avg_win"]:.2f}%')
    print(f'  avg_loss:        {results["trading_avg_loss"]:.2f}%')

    print('\n' + '=' * 80)
    print('Done!')

finally:
    dc.params = original_params
