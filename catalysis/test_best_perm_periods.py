#!/usr/bin/env python3
"""
Test best permutation (Perm 1) across different 3-month test periods in 2023-2024
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')
sys.path.insert(0, '/Users/beyondsyntax/Loop')

import loop
import polars as pl
from datetime import datetime, timedelta, timezone
import directional_conditional as dc
from loop.sfm.lightgbm.utils.tradeline_long_binary import apply_long_only_exit_strategy, calculate_atr
import numpy as np

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

# Trading parameters
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

# Define 3-month test periods for 2023-2024
test_periods = [
    # 2023
    ('2023-Q1', '2023-01-01', '2023-03-31'),
    ('2023-Q2', '2023-04-01', '2023-06-30'),
    ('2023-Q3', '2023-07-01', '2023-09-30'),
    ('2023-Q4', '2023-10-01', '2023-12-31'),
    # 2024
    ('2024-Q1', '2024-01-01', '2024-03-31'),
    ('2024-Q2', '2024-04-01', '2024-06-30'),
    ('2024-Q3', '2024-07-01', '2024-09-30'),
    ('2024-Q4', '2024-10-01', '2024-12-31'),
]

print('=' * 80)
print('TESTING BEST PERMUTATION ACROSS DIFFERENT 3-MONTH PERIODS')
print('=' * 80)
print('\nBest Parameters:')
for k, v in best_params.items():
    print(f'  {k:25s}: {v}')

# Load all data once at the start
print('\n' + '=' * 80)
print('LOADING ALL DATA')
print('=' * 80)

# Find earliest date needed (2023-01-01 minus 17 months)
earliest_date = datetime.strptime('2023-01-01', '%Y-%m-%d') - timedelta(days=17 * 30)
latest_date = datetime.strptime('2024-12-31', '%Y-%m-%d')

print(f'\nLoading data month by month from {earliest_date.strftime("%Y-%m-%d")} onwards')

# Load month by month with actual API calls
all_data_frames = []
current_date = earliest_date

while current_date <= datetime.now():
    month_str = current_date.strftime('%Y-%m')
    print(f'  Loading {month_str}...', end=' ', flush=True)

    # Load this month's data from API
    historical = loop.HistoricalData()
    historical.get_spot_klines(
        kline_size=300,
        start_date_limit=current_date.strftime('%Y-%m-%d'),
        n_rows=10000  # Limit rows per month
    )

    if len(historical.data) > 0:
        # Filter to only this month's data
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

# Combine all months
print(f'\nCombining all monthly data...')
all_data = pl.concat(all_data_frames)

print(f'Total loaded: {len(all_data):,} candles')
print(f'Date range: {all_data["datetime"].min()} to {all_data["datetime"].max()}')

results_list = []

for period_name, test_start, test_end in test_periods:
    print('\n' + '=' * 80)
    print(f'TESTING PERIOD: {period_name} ({test_start} to {test_end})')
    print('=' * 80)

    try:
        # Calculate data range: need 17 months before test period for train/val
        test_start_dt = datetime.strptime(test_start, '%Y-%m-%d')
        test_end_dt = datetime.strptime(test_end, '%Y-%m-%d')
        train_start_dt = test_start_dt - timedelta(days=17 * 30)  # ~17 months before test

        print(f'  Using data from {train_start_dt.strftime("%Y-%m-%d")} to {test_end_dt.strftime("%Y-%m-%d")}')

        # Slice the pre-loaded data
        full_data = all_data.filter(
            (pl.col('datetime') >= pl.lit(train_start_dt.replace(tzinfo=timezone.utc))) &
            (pl.col('datetime') <= pl.lit(test_end_dt.replace(tzinfo=timezone.utc)))
        )

        if len(full_data) < 100:
            print(f'  SKIPPED: Insufficient data ({len(full_data)} candles)')
            continue

        print(f'  Sliced {len(full_data):,} candles')

        # Prep data
        print('  Preparing data...')
        data_dict = dc.prep(full_data, round_params=best_params)

        # Train model
        print('  Training models...')
        results = dc.model(data_dict, round_params=best_params)

        # Get predictions
        y_pred = results['_preds']
        probabilities = results['extras']['probabilities']
        y_proba = probabilities['safe_long_given_movement']

        # Create OHLCV DataFrame for trading
        df = full_data.clone()
        df = calculate_atr(df, period=24)

        # Get test set (last 15%)
        n_total = len(df)
        val_end = int(n_total * 0.85)
        test_df = df[val_end:].select(['datetime', 'open', 'high', 'low', 'close', 'volume', 'atr_pct'])
        test_df = test_df.with_columns(pl.lit(0).alias('label'))

        # Align predictions
        if len(y_pred) != len(test_df):
            min_len = min(len(y_pred), len(test_df))
            test_df = test_df[:min_len]
            y_pred = y_pred[:min_len]
            y_proba = y_proba[:min_len]

        # Run trading simulation
        print(f'  Running trading simulation on {len(test_df)} candles...')
        _, trading_results = apply_long_only_exit_strategy(
            test_df, y_pred, y_proba, best_params['threshold_pct'], exit_config
        )

        # Store results
        period_results = {
            'period': period_name,
            'test_start': test_start,
            'test_end': test_end,
            'n_candles': len(test_df),
            'recall': results['recall'],
            'precision': results['precision'],
            'fpr': results['fpr'],
            'auc': results['auc'],
            'accuracy': results['accuracy'],
            'return_pct': trading_results['total_return_net_pct'],
            'win_rate': trading_results['trade_win_rate_pct'],
            'trades': trading_results['trades_count'],
            'avg_win_usd': trading_results['avg_win'],
            'avg_loss_usd': trading_results['avg_loss']
        }
        results_list.append(period_results)

        # Print results
        print(f'\n  Classification Metrics:')
        print(f'    recall:    {results["recall"]:.4f}')
        print(f'    precision: {results["precision"]:.4f}')
        print(f'    fpr:       {results["fpr"]:.4f}')
        print(f'    auc:       {results["auc"]:.4f}')
        print(f'    accuracy:  {results["accuracy"]:.4f}')
        print(f'\n  Trading Metrics:')
        print(f'    return:    {trading_results["total_return_net_pct"]:.2f}%')
        print(f'    win_rate:  {trading_results["trade_win_rate_pct"]:.2f}%')
        print(f'    trades:    {trading_results["trades_count"]:.0f}')
        print(f'    avg_win:   ${trading_results["avg_win"]:.2f}')
        print(f'    avg_loss:  ${trading_results["avg_loss"]:.2f}')

    except Exception as e:
        print(f'  ERROR: {str(e)}')
        continue

# Summary
print('\n' + '=' * 80)
print('SUMMARY ACROSS ALL PERIODS')
print('=' * 80)

if results_list:
    results_df = pl.DataFrame(results_list)
    print(results_df)

    # Save to CSV
    output_file = '/Users/beyondsyntax/Loop/catalysis/best_perm_period_test_results.csv'
    results_df.write_csv(output_file)
    print(f'\nResults saved to: {output_file}')

    # Calculate averages
    print('\n' + '=' * 80)
    print('AVERAGE METRICS ACROSS PERIODS:')
    print('=' * 80)
    print(f'\nClassification:')
    print(f'  avg_auc:       {results_df["auc"].mean():.4f}')
    print(f'  avg_recall:    {results_df["recall"].mean():.4f}')
    print(f'  avg_precision: {results_df["precision"].mean():.4f}')
    print(f'  avg_accuracy:  {results_df["accuracy"].mean():.4f}')
    print(f'\nTrading:')
    print(f'  avg_return:    {results_df["return_pct"].mean():.2f}%')
    print(f'  avg_win_rate:  {results_df["win_rate"].mean():.2f}%')
    print(f'  avg_trades:    {results_df["trades"].mean():.1f}')
    print(f'  std_return:    {results_df["return_pct"].std():.2f}%')
else:
    print('\nNo results to display')

print('\n' + '=' * 80)
print('Done!')
