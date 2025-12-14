#!/usr/bin/env python3
"""
Test Directional Conditional Probability Model Across Multiple Time Periods

Tests on different 3-month periods in 2023 and 2024 to check consistency.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')

import loop
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import directional_conditional as dc

# Define test periods (end dates for 20-month data loads)
# Test period will be last 3 months of each load
TEST_PERIODS = [
    ('Q1 2023', '2023-03-31'),  # Test: Jan-Mar 2023
    ('Q2 2023', '2023-06-30'),  # Test: Apr-Jun 2023
    ('Q3 2023', '2023-09-30'),  # Test: Jul-Sep 2023
    ('Q4 2023', '2023-12-31'),  # Test: Oct-Dec 2023
    ('Q1 2024', '2024-03-31'),  # Test: Jan-Mar 2024
    ('Q2 2024', '2024-06-30'),  # Test: Apr-Jun 2024
    ('Q3 2024', '2024-09-30'),  # Test: Jul-Sep 2024
    ('Q4 2024', '2024-11-17'),  # Test: Aug-Nov 2024 (recent)
]

print('=' * 80)
print('DIRECTIONAL CONDITIONAL MODEL - MULTI-PERIOD TEST')
print('=' * 80)
print(f'\nTesting {len(TEST_PERIODS)} different 3-month periods')
print('Each test uses 20 months of data with 70/15/15 train/val/test split')
print('\nPeriods to test:')
for name, end_date in TEST_PERIODS:
    print(f'  {name}: Test period ending {end_date}')

# Load ALL data once
print('\n' + '=' * 80)
print('LOADING ALL DATA (ONE TIME)')
print('=' * 80)

# Load from 20 months before now (guaranteed to have data)
kline_size = 300
end_date = datetime.now()
start_date = end_date - timedelta(days=20 * 30)
start_date_str = start_date.strftime('%Y-%m-%d')

print(f'\nLoading 20 months of data from {start_date_str}...')
print(f'This will cover all test periods that have available data')
print(f'Fetching {kline_size}s candles from database...', flush=True)

historical = loop.HistoricalData()
historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
all_data = historical.data

print(f'\nLoaded {len(all_data):,} candles total')
print(f'Date range: {all_data["datetime"].min()} to {all_data["datetime"].max()}')

results = []

print('\n' + '=' * 80)
print('TRAINING MODELS FOR EACH PERIOD')
print('=' * 80)

for period_name, period_end_str in TEST_PERIODS:
    print('\n' + '=' * 80)
    print(f'TESTING PERIOD: {period_name}')
    print('=' * 80)

    # Calculate data range for this period
    period_end = datetime.strptime(period_end_str, '%Y-%m-%d')
    period_start = period_end - timedelta(days=20 * 30)

    print(f'\nFiltering data: {period_start.strftime("%Y-%m-%d")} to {period_end_str}')

    # Filter pre-loaded data to this period's range
    period_start_str = period_start.strftime("%Y-%m-%d")
    full_data = all_data.filter(
        (pl.col('datetime').dt.strftime("%Y-%m-%d") >= period_start_str) &
        (pl.col('datetime').dt.strftime("%Y-%m-%d") <= period_end_str)
    )

    print(f'Filtered to {len(full_data):,} candles for this period')

    if len(full_data) < 10000:
        print(f'WARNING: Not enough data for period {period_name}, skipping...')
        continue

    # Train with UEL
    print(f'\nTraining 3 directional models (LONG, SHORT, MOVEMENT)...')

    uel = loop.UniversalExperimentLoop(
        data=full_data,
        single_file_model=dc
    )

    print('  Running UEL experiment...')
    uel.run(
        experiment_name=f'directional_conditional_{period_name.replace(" ", "_")}',
        n_permutations=1,
        random_search=False
    )
    print('  Training complete!')

    # Extract results
    log = uel.experiment_log
    extras = uel.extras[0]

    # Get test period dates
    alignment = uel._alignment[0]
    first_test_dt = alignment['first_test_datetime']
    last_test_dt = alignment['last_test_datetime']

    print(f'Test period: {first_test_dt} to {last_test_dt}')

    # Extract key metrics
    probs = extras['probabilities']
    class_dists = extras['class_distributions']

    # Calculate statistics
    p_long = probs['long']
    p_short = probs['short']
    p_movement = probs['movement']
    p_long_given_movement = probs['long_given_movement']
    p_short_given_movement = probs['short_given_movement']

    # Strong signals
    strong_long = np.sum((p_long_given_movement > 0.7) & (p_movement > 0.3))
    strong_short = np.sum((p_short_given_movement > 0.7) & (p_movement > 0.3))
    low_movement = np.sum(p_movement < 0.2)

    # Correlations
    corr_long = np.corrcoef(p_long, p_long_given_movement)[0, 1]
    corr_short = np.corrcoef(p_short, p_short_given_movement)[0, 1]

    # Big changes
    big_changes = np.sum(np.abs(p_long - p_long_given_movement) > 0.3)

    # Store results
    period_results = {
        'period_name': period_name,
        'test_start': first_test_dt,
        'test_end': last_test_dt,
        'n_samples': len(p_long),
        'accuracy': log['accuracy'][0],
        'precision': log['precision'][0],
        'recall': log['recall'][0],
        'mean_p_long': np.mean(p_long),
        'mean_p_short': np.mean(p_short),
        'mean_p_movement': np.mean(p_movement),
        'mean_p_long_given_movement': np.mean(p_long_given_movement),
        'mean_p_short_given_movement': np.mean(p_short_given_movement),
        'strong_long_signals': strong_long,
        'strong_short_signals': strong_short,
        'low_movement_pct': low_movement / len(p_movement) * 100,
        'corr_long': corr_long,
        'corr_short': corr_short,
        'big_changes_pct': big_changes / len(p_long) * 100,
        'long_positive_pct': class_dists['long'].get(1, 0) / sum(class_dists['long'].values()) * 100,
        'short_positive_pct': class_dists['short'].get(1, 0) / sum(class_dists['short'].values()) * 100,
        'movement_positive_pct': class_dists['movement'].get(1, 0) / sum(class_dists['movement'].values()) * 100
    }

    results.append(period_results)

    print(f'\nQuick Summary:')
    print(f'  Strong LONG signals:  {strong_long} ({strong_long/len(p_long)*100:.1f}%)')
    print(f'  Strong SHORT signals: {strong_short} ({strong_short/len(p_short)*100:.1f}%)')
    print(f'  Mean P(MOVEMENT):     {np.mean(p_movement):.3f}')
    print(f'  Mean P(LONG|movement): {np.mean(p_long_given_movement):.3f}')
    print(f'  Mean P(SHORT|movement): {np.mean(p_short_given_movement):.3f}')

# Print comprehensive comparison
print('\n' + '=' * 80)
print('COMPREHENSIVE RESULTS ACROSS ALL PERIODS')
print('=' * 80)

if len(results) == 0:
    print('No results to display')
else:
    # Convert to DataFrame for easy viewing
    results_df = pl.DataFrame(results)

    print('\n--- MODEL PERFORMANCE ---')
    print(results_df.select(['period_name', 'accuracy', 'precision', 'recall']))

    print('\n--- RAW PROBABILITIES (mean) ---')
    print(results_df.select(['period_name', 'mean_p_long', 'mean_p_short', 'mean_p_movement']))

    print('\n--- CONDITIONAL PROBABILITIES (mean) ---')
    print(results_df.select(['period_name', 'mean_p_long_given_movement', 'mean_p_short_given_movement']))

    print('\n--- SIGNAL COUNTS ---')
    print(results_df.select(['period_name', 'strong_long_signals', 'strong_short_signals']))

    print('\n--- CORRELATIONS (raw vs conditional) ---')
    print(results_df.select(['period_name', 'corr_long', 'corr_short']))

    print('\n--- CLASS DISTRIBUTIONS (% positive) ---')
    print(results_df.select(['period_name', 'long_positive_pct', 'short_positive_pct', 'movement_positive_pct']))

    print('\n--- MARKET CHARACTERISTICS ---')
    print(results_df.select(['period_name', 'low_movement_pct', 'big_changes_pct']))

    # Calculate statistics across periods
    print('\n' + '=' * 80)
    print('STATISTICS ACROSS ALL PERIODS')
    print('=' * 80)

    print('\nP(LONG | movement):')
    print(f'  Mean:   {results_df["mean_p_long_given_movement"].mean():.3f}')
    print(f'  Std:    {results_df["mean_p_long_given_movement"].std():.3f}')
    print(f'  Min:    {results_df["mean_p_long_given_movement"].min():.3f}')
    print(f'  Max:    {results_df["mean_p_long_given_movement"].max():.3f}')

    print('\nP(SHORT | movement):')
    print(f'  Mean:   {results_df["mean_p_short_given_movement"].mean():.3f}')
    print(f'  Std:    {results_df["mean_p_short_given_movement"].std():.3f}')
    print(f'  Min:    {results_df["mean_p_short_given_movement"].min():.3f}')
    print(f'  Max:    {results_df["mean_p_short_given_movement"].max():.3f}')

    print('\nP(MOVEMENT):')
    print(f'  Mean:   {results_df["mean_p_movement"].mean():.3f}')
    print(f'  Std:    {results_df["mean_p_movement"].std():.3f}')
    print(f'  Min:    {results_df["mean_p_movement"].min():.3f}')
    print(f'  Max:    {results_df["mean_p_movement"].max():.3f}')

    print('\nStrong Signals:')
    print(f'  LONG  - Mean: {results_df["strong_long_signals"].mean():.0f}, Range: {results_df["strong_long_signals"].min()}-{results_df["strong_long_signals"].max()}')
    print(f'  SHORT - Mean: {results_df["strong_short_signals"].mean():.0f}, Range: {results_df["strong_short_signals"].min()}-{results_df["strong_short_signals"].max()}')

    # Save results
    output_file = '/Users/beyondsyntax/Loop/catalysis/directional_periods_results.csv'
    results_df.write_csv(output_file)
    print(f'\n\nResults saved to: {output_file}')

print('\n' + '=' * 80)
print('Done!')
