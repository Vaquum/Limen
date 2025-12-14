#!/usr/bin/env python3
"""
Run best permutation (Perm 1) to get full UEL metrics including trading performance
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
from loop.sfm.lightgbm.utils.tradeline_long_binary import apply_long_only_exit_strategy

print('=' * 80)
print('RUNNING BEST PERMUTATION (Perm 1) FOR FULL METRICS')
print('=' * 80)

# Load 20 months of data
print('\nLoading 20 months of data...')
kline_size = 300
end_date = datetime.now()
n_months = 20
start_date = end_date - timedelta(days=n_months * 30)
start_date_str = start_date.strftime('%Y-%m-%d')

historical = loop.HistoricalData()
historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
full_data = historical.data

print(f'Loaded {len(full_data):,} candles')
print(f'Date range: {full_data["datetime"].min()} to {full_data["datetime"].max()}')

# Best permutation parameters
print('\n' + '=' * 80)
print('BEST PERMUTATION PARAMETERS:')
print('=' * 80)
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

for k, v in best_params.items():
    print(f'  {k:25s}: {v}')

# Create custom params function that returns only this permutation
def custom_params():
    return {k: [v] for k, v in best_params.items()}

# Temporarily replace params
original_params = dc.params
dc.params = custom_params

print('\n' + '=' * 80)
print('RUNNING UEL WITH BEST PERMUTATION')
print('=' * 80)

try:
    uel = loop.UniversalExperimentLoop(
        data=full_data,
        single_file_model=dc
    )

    uel.run(
        experiment_name='best_directional_perm1',
        n_permutations=1,
        random_search=False
    )

    results = uel.experiment_log

    print('\n' + '=' * 80)
    print('FULL METRICS FOR BEST PERMUTATION:')
    print('=' * 80)

    # Display all columns
    for col in results.columns:
        val = results[col][0]
        print(f'{col:30s}: {val}')

    # Save results
    print('\n' + '=' * 80)
    print('SAVING RESULTS')
    print('=' * 80)

    # Drop object columns for CSV
    cols_to_keep = [col for col in results.columns if results[col].dtype != pl.Object]
    results_clean = results.select(cols_to_keep)
    results_clean.write_csv('/Users/beyondsyntax/Loop/catalysis/best_perm1_full_metrics.csv')

    print('\nResults saved to: best_perm1_full_metrics.csv')

finally:
    dc.params = original_params

print('\n' + '=' * 80)
print('Done!')
