#!/usr/bin/env python3
"""
Compare Directional Conditional Model vs Baseline tradeline_long_binary
SKIPPING PERMUTATION 38

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
import numpy as np
import traceback

print('=' * 80)
print('DIRECTIONAL CONDITIONAL vs BASELINE COMPARISON (SKIP PERMUTATION 38)')
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

# Test 2: Directional Conditional (49 permutations, skipping #38)
print('\n' + '=' * 80)
print('TEST 2: DIRECTIONAL CONDITIONAL (49 permutations, SKIPPING #38)')
print('=' * 80)

# Generate all 50 permutations manually
np.random.seed(42)
params = dc.params()
param_names = list(params.keys())

all_permutations = []
for i in range(50):
    perm = {}
    for name in param_names:
        values = params[name]
        perm[name] = np.random.choice(values)
    all_permutations.append(perm)

# Remove permutation 38 (index 37)
print(f'\nSkipping permutation 38 with params:')
skipped = all_permutations[37]
for k, v in skipped.items():
    if k in ['threshold_pct', 'lookahead_hours', 'use_safer', 'conditional_threshold', 'movement_threshold']:
        print(f'  {k}: {v}')

permutations_to_run = all_permutations[:37] + all_permutations[38:]

# BATCH MODE: Run in batches of 5 to allow resuming on crash
BATCH_SIZE = 5
START_BATCH = 0  # Change this to resume from a specific batch (0-indexed)

total_permutations = len(permutations_to_run)
total_batches = (total_permutations + BATCH_SIZE - 1) // BATCH_SIZE

print(f'\n*** BATCH MODE: Running {total_permutations} permutations in batches of {BATCH_SIZE} ({total_batches} batches total) ***')
print(f'*** Starting from batch {START_BATCH + 1} ***')

# Run each permutation manually
results = []
for batch_num in range(START_BATCH, total_batches):
    batch_start = batch_num * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, total_permutations)

    print(f'\n{"=" * 80}')
    print(f'BATCH {batch_num + 1}/{total_batches}: Permutations {batch_start + 1}-{batch_end}')
    print(f'{"=" * 80}')

    batch_results = []
    for idx in range(batch_start, batch_end):
        perm = permutations_to_run[idx]
        actual_perm_num = idx + 1 if idx < 37 else idx + 2
        print(f'\n[{idx+1}/{total_permutations}] Running permutation {actual_perm_num}...')

        uel_directional = loop.UniversalExperimentLoop(
            data=full_data,
            single_file_model=dc
        )

        # Create a custom params function that returns only this permutation
        def custom_params():
            return {k: [v] for k, v in perm.items()}

        # Temporarily replace params
        original_params = dc.params
        dc.params = custom_params

        try:
            uel_directional.run(
                experiment_name=f'directional_conditional_perm_{actual_perm_num}',
                n_permutations=1,
                random_search=False
            )

            perm_log = uel_directional.experiment_log
            perm_log = perm_log.with_columns(pl.lit(actual_perm_num).alias('original_permutation'))
            batch_results.append(perm_log)
            results.append(perm_log)
        except Exception as e:
            print(f'ERROR in permutation {actual_perm_num}: {e}')
            traceback.print_exc()
        finally:
            dc.params = original_params

    # Save batch results incrementally
    if batch_results:
        batch_log = pl.concat(batch_results)
        batch_cols_to_keep = [col for col in batch_log.columns if batch_log[col].dtype != pl.Object]
        batch_log_clean = batch_log.select(batch_cols_to_keep)
        batch_log_clean.write_csv(f'/Users/beyondsyntax/Loop/catalysis/directional_batch_{batch_num+1}_results.csv')
        print(f'\nBatch {batch_num+1} saved to directional_batch_{batch_num+1}_results.csv')
    else:
        print(f'\nBatch {batch_num+1} had no successful results')

# Combine all results
if results:
    directional_log = pl.concat(results)

    print('\n' + '=' * 80)
    print('DIRECTIONAL CONDITIONAL RESULTS (top 10 by AUC)')
    print('=' * 80)
    print(directional_log[['accuracy', 'precision', 'recall', 'auc', 'threshold_pct', 'lookahead_hours', 'conditional_threshold', 'movement_threshold', 'use_safer', 'original_permutation']].sort('auc', descending=True).head(10))

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
        'use_safer': best_directional_row['use_safer'][0],
        'original_permutation': best_directional_row['original_permutation'][0]
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
    print(f'  Permutation: {best_directional["original_permutation"]}')

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

    # Drop object dtypes for CSV compatibility (cannot be serialized)
    baseline_cols_to_keep = [col for col in baseline_log.columns if baseline_log[col].dtype != pl.Object]
    directional_cols_to_keep = [col for col in directional_log.columns if directional_log[col].dtype != pl.Object]

    baseline_log_clean = baseline_log.select(baseline_cols_to_keep)
    directional_log_clean = directional_log.select(directional_cols_to_keep)

    baseline_log_clean.write_csv('/Users/beyondsyntax/Loop/catalysis/baseline_tradeline_results_skip38.csv')
    directional_log_clean.write_csv('/Users/beyondsyntax/Loop/catalysis/directional_conditional_results_skip38.csv')

    print('\nResults saved to:')
    print('  baseline_tradeline_results_skip38.csv')
    print('  directional_conditional_results_skip38.csv')

    # Save summary report to file
    with open('/Users/beyondsyntax/Loop/catalysis/comparison_summary_skip38.txt', 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('DIRECTIONAL CONDITIONAL vs BASELINE COMPARISON (SKIP PERMUTATION 38)\n')
        f.write('=' * 80 + '\n\n')

        f.write('BASELINE RESULTS:\n')
        f.write(f'  Accuracy:  {best_baseline["accuracy"]:.4f}\n')
        f.write(f'  Precision: {best_baseline["precision"]:.4f}\n')
        f.write(f'  Recall:    {best_baseline["recall"]:.4f}\n')
        f.write(f'  FPR:       {best_baseline["fpr"]:.4f}\n')
        f.write(f'  AUC:       {best_baseline["auc"]:.4f}\n\n')

        f.write('BEST DIRECTIONAL (from 49 permutations):\n')
        f.write(f'  Accuracy:  {best_directional["accuracy"]:.4f}\n')
        f.write(f'  Precision: {best_directional["precision"]:.4f}\n')
        f.write(f'  Recall:    {best_directional["recall"]:.4f}\n')
        f.write(f'  FPR:       {best_directional["fpr"]:.4f}\n')
        f.write(f'  AUC:       {best_directional["auc"]:.4f}\n')
        f.write(f'  Conditional threshold: {best_directional["conditional_threshold"]:.2f}\n')
        f.write(f'  Using safer: {best_directional["use_safer"]}\n')
        f.write(f'  Permutation: {best_directional["original_permutation"]}\n\n')

        f.write('IMPROVEMENT:\n')
        f.write(f'  Accuracy:  {(best_directional["accuracy"] - best_baseline["accuracy"]):.4f} ({(best_directional["accuracy"]/best_baseline["accuracy"]-1)*100:+.1f}%)\n')
        f.write(f'  Precision: {(best_directional["precision"] - best_baseline["precision"]):.4f} ({(best_directional["precision"]/best_baseline["precision"]-1)*100:+.1f}%)\n')
        f.write(f'  Recall:    {(best_directional["recall"] - best_baseline["recall"]):.4f} ({(best_directional["recall"]/best_baseline["recall"]-1)*100:+.1f}%)\n')
        if best_baseline["fpr"] > 0:
            fpr_pct = (best_directional["fpr"]/best_baseline["fpr"]-1)*100
            f.write(f'  FPR:       {fpr_diff:.4f} ({fpr_pct:+.1f}%)\n')
        else:
            f.write(f'  FPR:       {fpr_diff:.4f} (baseline FPR=0, directional FPR={best_directional["fpr"]:.4f})\n')
        f.write(f'  AUC:       {(best_directional["auc"] - best_baseline["auc"]):.4f} ({(best_directional["auc"]/best_baseline["auc"]-1)*100:+.1f}%)\n\n')

        f.write('TOP 10 PERMUTATIONS BY AUC:\n')
        f.write('=' * 80 + '\n')
        top_10 = directional_log.sort('auc', descending=True).head(10)
        f.write(str(top_10[['accuracy', 'precision', 'recall', 'auc', 'conditional_threshold', 'use_safer', 'original_permutation']]))

    print('  comparison_summary_skip38.txt')
else:
    print('\nNo results collected!')

print('\n' + '=' * 80)
print('Done!')
