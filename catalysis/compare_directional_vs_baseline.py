#!/usr/bin/env python3
"""
Compare Directional Conditional Model vs Baseline tradeline_long_binary

Tests both models on the same data and compares performance.
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
from loop.sfm.lightgbm import tradeline_long_binary as baseline

print('=' * 80)
print('DIRECTIONAL CONDITIONAL vs BASELINE COMPARISON')
print('=' * 80)

# Load all data in one go, but show month-by-month progress
print('\nLoading 20 months of data with month-by-month logging...')
kline_size = 300
end_date = datetime.now()
n_months = 20

# Calculate start date for full load
start_date = end_date - timedelta(days=n_months * 30)
start_date_str = start_date.strftime('%Y-%m-%d')

print(f'Loading all data from {start_date_str} in one DB query...')
historical = loop.HistoricalData()
historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
full_data = historical.data

print(f'\nLoaded {len(full_data):,} total candles')
print(f'Date range: {full_data["datetime"].min()} to {full_data["datetime"].max()}')

# Show month-by-month breakdown
print('\nMonth-by-month breakdown:')
for i in range(n_months, 0, -1):
    month_end = end_date - timedelta(days=(i-1) * 30)
    month_start = end_date - timedelta(days=i * 30)
    month_start_str = month_start.strftime('%Y-%m-%d')
    month_end_str = month_end.strftime('%Y-%m-%d')

    # Filter to this month
    month_data = full_data.filter(
        (pl.col('datetime').dt.strftime("%Y-%m-%d") >= month_start_str) &
        (pl.col('datetime').dt.strftime("%Y-%m-%d") <= month_end_str)
    )

    print(f'  Month {n_months-i+1}/{n_months} ({month_start_str} to {month_end_str}): {len(month_data):,} candles')

# Test 1: Baseline (tradeline_long_binary)
print('\n' + '=' * 80)
print('TEST 1: BASELINE (tradeline_long_binary)')
print('=' * 80)

uel_baseline = loop.UniversalExperimentLoop(
    data=full_data,
    single_file_model=baseline
)

uel_baseline.run(
    experiment_name='baseline_tradeline_long',
    n_permutations=1,
    random_search=False
)

baseline_log = uel_baseline.experiment_log
print('\nBaseline Results:')
print(baseline_log[['accuracy', 'precision', 'recall', 'fpr', 'auc']])

# Test 2: Directional Conditional (default params)
print('\n' + '=' * 80)
print('TEST 2: DIRECTIONAL CONDITIONAL (P(LONG | movement) > 0.7)')
print('=' * 80)

uel_directional = loop.UniversalExperimentLoop(
    data=full_data,
    single_file_model=dc
)

uel_directional.run(
    experiment_name='directional_conditional_sweep',
    n_permutations=50,  # Test 50 random combinations from the expanded parameter grid
    random_search=True
)

directional_log = uel_directional.experiment_log
print('\nDirectional Conditional Results (top 10 by AUC):')
print(directional_log[['accuracy', 'precision', 'recall', 'auc', 'threshold_pct', 'lookahead_hours', 'conditional_threshold', 'movement_threshold', 'use_safer']].head(10))

# Compare best results
print('\n' + '=' * 80)
print('COMPARISON SUMMARY')
print('=' * 80)

# Get best results as dicts with scalar values
best_baseline_row = baseline_log.sort('auc', descending=True).head(1)
best_directional_row = directional_log.sort('auc', descending=True).head(1)

# Extract scalar values
best_baseline = {
    'accuracy': best_baseline_row['accuracy'][0],
    'precision': best_baseline_row['precision'][0],
    'recall': best_baseline_row['recall'][0],
    'fpr': best_baseline_row['fpr'][0],
    'auc': best_baseline_row['auc'][0]
}

best_directional = {
    'accuracy': best_directional_row['accuracy'][0],
    'precision': best_directional_row['precision'][0],
    'recall': best_directional_row['recall'][0],
    'fpr': best_directional_row['fpr'][0],
    'auc': best_directional_row['auc'][0],
    'conditional_threshold': best_directional_row['conditional_threshold'][0],
    'use_safer': best_directional_row['use_safer'][0]
}

print('\nBest Baseline:')
print(f'  Accuracy:  {best_baseline["accuracy"]:.4f}')
print(f'  Precision: {best_baseline["precision"]:.4f}')
print(f'  Recall:    {best_baseline["recall"]:.4f}')
print(f'  FPR:       {best_baseline["fpr"]:.4f}')
print(f'  AUC:       {best_baseline["auc"]:.4f}')

print('\nBest Directional:')
print(f'  Accuracy:  {best_directional["accuracy"]:.4f}')
print(f'  Precision: {best_directional["precision"]:.4f}')
print(f'  Recall:    {best_directional["recall"]:.4f}')
print(f'  FPR:       {best_directional["fpr"]:.4f}')
print(f'  AUC:       {best_directional["auc"]:.4f}')
print(f'  Conditional threshold: {best_directional["conditional_threshold"]:.2f}')
print(f'  Using safer: {best_directional["use_safer"]}')

print('\nImprovement:')
print(f'  Accuracy:  {(best_directional["accuracy"] - best_baseline["accuracy"]):.4f} ({(best_directional["accuracy"]/best_baseline["accuracy"]-1)*100:+.1f}%)')
print(f'  Precision: {(best_directional["precision"] - best_baseline["precision"]):.4f} ({(best_directional["precision"]/best_baseline["precision"]-1)*100:+.1f}%)')
print(f'  Recall:    {(best_directional["recall"] - best_baseline["recall"]):.4f} ({(best_directional["recall"]/best_baseline["recall"]-1)*100:+.1f}%)')

# Handle FPR calculation (baseline FPR might be 0)
fpr_diff = best_directional["fpr"] - best_baseline["fpr"]
if best_baseline["fpr"] > 0:
    fpr_pct = (best_directional["fpr"]/best_baseline["fpr"]-1)*100
    print(f'  FPR:       {fpr_diff:.4f} ({fpr_pct:+.1f}%)')
else:
    print(f'  FPR:       {fpr_diff:.4f} (baseline FPR=0, directional FPR={best_directional["fpr"]:.4f})')

print(f'  AUC:       {(best_directional["auc"] - best_baseline["auc"]):.4f} ({(best_directional["auc"]/best_baseline["auc"]-1)*100:+.1f}%)')

# Save comparison results
print('\n' + '=' * 80)
print('SAVING RESULTS')
print('=' * 80)

baseline_log.write_csv('/Users/beyondsyntax/Loop/catalysis/baseline_tradeline_results.csv')
directional_log.write_csv('/Users/beyondsyntax/Loop/catalysis/directional_conditional_results.csv')

print('\nResults saved to:')
print('  baseline_tradeline_results.csv')
print('  directional_conditional_results.csv')

print('\n' + '=' * 80)
print('Done!')
