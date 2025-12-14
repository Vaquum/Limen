#!/usr/bin/env python3
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

print('Loading data...')
kline_size = 300
end_date = datetime.now()
start_date = end_date - timedelta(days=20 * 30)
historical = loop.HistoricalData()
historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date.strftime('%Y-%m-%d'))
full_data = historical.data

best_params = {
    'threshold_pct': 0.005, 'lookahead_hours': 72, 'quantile_threshold': 0.7,
    'min_height_pct': 0.005, 'max_duration_hours': 48, 'conditional_threshold': 0.6,
    'movement_threshold': 0.3, 'use_safer': True, 'n_estimators': 300,
    'num_leaves': 63, 'learning_rate': 0.05, 'max_depth': -1,
    'min_child_samples': 20, 'feature_fraction': 0.9, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'lambda_l1': 0, 'lambda_l2': 0
}

print('Prepping data...')
data_dict = dc.prep(full_data, round_params=best_params)

print('Training model...')
results = dc.model(data_dict, round_params=best_params)

print('Getting predictions...')
y_pred = results['_preds']
probabilities = results['extras']['probabilities']
y_proba = probabilities['safe_long_given_movement']

print('Creating OHLCV DataFrame for trading...')
# Get original data and calculate ATR
df = full_data.clone()
df = calculate_atr(df, period=24)

# Get test set based on data length
n_total = len(df)
train_end = int(n_total * 0.7)
val_end = int(n_total * 0.85)
test_df = df[val_end:].select(['datetime', 'open', 'high', 'low', 'close', 'volume', 'atr_pct'])
test_df = test_df.with_columns(pl.lit(0).alias('label'))

# Align predictions with test_df
if len(y_pred) != len(test_df):
    print(f'WARNING: Prediction length {len(y_pred)} != test_df length {len(test_df)}')
    min_len = min(len(y_pred), len(test_df))
    test_df = test_df[:min_len]
    y_pred = y_pred[:min_len]
    y_proba = y_proba[:min_len]

print(f'Running trading simulation on {len(test_df)} candles...')
exit_config = {
    'confidence_threshold': 0.5, 'position_size': 0.1, 'min_stop_loss': 0.01,
    'max_stop_loss': 0.05, 'atr_stop_multiplier': 2.0, 'trailing_activation': 0.01,
    'trailing_distance': 0.5, 'loser_timeout_hours': 12, 'max_hold_hours': 96,
    'default_atr_pct': 0.01, 'initial_capital': 100000.0
}

_, trading_results = apply_long_only_exit_strategy(
    test_df, y_pred, y_proba, best_params['threshold_pct'], exit_config
)

print('\n' + '=' * 80)
print('ALL METRICS:')
print('=' * 80)
print(f'\nClassification:')
print(f'  recall:      {results["recall"]:.4f}')
print(f'  precision:   {results["precision"]:.4f}')
print(f'  fpr:         {results["fpr"]:.4f}')
print(f'  auc:         {results["auc"]:.4f}')
print(f'  accuracy:    {results["accuracy"]:.4f}')
print(f'\nTrading:')
print(f'  return_pct:  {trading_results["total_return_net_pct"]:.2f}%')
print(f'  win_rate:    {trading_results["trade_win_rate_pct"]:.2f}%')
print(f'  trades:      {trading_results["trades_count"]:.0f}')
print(f'  avg_win:     ${trading_results["avg_win"]:.2f}')
print(f'  avg_loss:    ${trading_results["avg_loss"]:.2f}')
