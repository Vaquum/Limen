#!/usr/bin/env python3
"""
Test Multi-Threshold Resistance Finder

Trains 10 binary models for different breakout thresholds.
Analyzes probability gaps to identify ML-predicted resistance levels.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')

import loop
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import multi_threshold_resistance_finder as mtrf

print('=' * 80)
print('MULTI-THRESHOLD RESISTANCE FINDER')
print('=' * 80)

# Load data
print('\nLoading 20 months of data...')
kline_size = 300
end_date = datetime.now()
start_date = end_date - timedelta(days=20 * 30)
start_date_str = start_date.strftime('%Y-%m-%d')

historical = loop.HistoricalData()
historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
full_data = historical.data
print(f'Loaded {len(full_data):,} candles')

# Use UEL to train
print('\nTraining multi-threshold models...')
print(f'Thresholds: {[f"{t*100:.1f}%" for t in mtrf.THRESHOLDS]}')

uel = loop.UniversalExperimentLoop(
    data=full_data,
    single_file_model=mtrf
)

uel.run(
    experiment_name='resistance_finder_test',
    n_permutations=1,
    random_search=False
)

print('\n' + '=' * 80)
print('RESISTANCE ANALYSIS')
print('=' * 80)

# Extract results
log = uel.experiment_log
extras = uel.extras[0]

test_probabilities = extras['test_probabilities']
resistance_analysis = extras['resistance_analysis']

print('\nAverage Probability Deltas (across all test samples):')
print('-' * 80)

for delta_info in resistance_analysis['avg_deltas']:
    low_pct = delta_info['threshold_low'] * 100
    high_pct = delta_info['threshold_high'] * 100
    avg_delta = delta_info['avg_delta']
    max_delta = delta_info['max_delta']

    marker = ' ← RESISTANCE ZONE' if delta_info in resistance_analysis['resistance_zones'] else ''

    print(f'{low_pct:.1f}% → {high_pct:.1f}%: Δ={avg_delta:.4f} (max={max_delta:.4f}){marker}')

print('\n' + '-' * 80)
print(f'Resistance Threshold: {resistance_analysis["resistance_threshold"]:.4f}')
print(f'Identified {len(resistance_analysis["resistance_zones"])} resistance zones')

if len(resistance_analysis['resistance_zones']) > 0:
    print('\nResistance Zones:')
    for zone in resistance_analysis['resistance_zones']:
        low_pct = zone['threshold_low'] * 100
        high_pct = zone['threshold_high'] * 100
        print(f'  {low_pct:.1f}% - {high_pct:.1f}%: Δ={zone["avg_delta"]:.4f}')

# Show example probabilities for first few test samples
print('\n' + '=' * 80)
print('EXAMPLE: Probability Curves for First 5 Test Samples')
print('=' * 80)

n_examples = min(5, len(test_probabilities[mtrf.THRESHOLDS[0]]))

for sample_idx in range(n_examples):
    print(f'\nSample {sample_idx + 1}:')
    print('  Threshold  | Probability | Delta from previous')
    print('  ' + '-' * 50)

    prev_prob = None
    for threshold in mtrf.THRESHOLDS:
        prob = test_probabilities[threshold][sample_idx]
        delta_str = f'{prev_prob - prob:.4f}' if prev_prob is not None else '    -'

        # Mark large deltas
        marker = ' ← BIG DROP' if prev_prob is not None and (prev_prob - prob) > 0.15 else ''

        print(f'  {threshold*100:5.1f}%    | {prob:.4f}      | {delta_str}{marker}')
        prev_prob = prob

# Summary statistics
print('\n' + '=' * 80)
print('SUMMARY STATISTICS')
print('=' * 80)

print('\nAverage probability for each threshold (across all test samples):')
for threshold in mtrf.THRESHOLDS:
    avg_prob = np.mean(test_probabilities[threshold])
    print(f'  P(breakout > {threshold*100:.1f}%) = {avg_prob:.4f}')

print('\nModel Performance (for 0.5% threshold):')
print(f'  Accuracy: {log["accuracy"][0]:.4f}')
print(f'  Precision: {log["precision"][0]:.4f}')
print(f'  Recall: {log["recall"][0]:.4f}')
# F1 and AUC may not be in log for custom multi-threshold setup
if 'f1' in log.columns:
    print(f'  F1: {log["f1"][0]:.4f}')
if 'roc_auc' in log.columns:
    print(f'  AUC: {log["roc_auc"][0]:.4f}')

# Trading application examples
print('\n' + '=' * 80)
print('POTENTIAL TRADING APPLICATIONS')
print('=' * 80)

print("""
1. Take-Profit Targets:
   - Set TP just below identified resistance levels
   - If resistance is at 2.5-3.0%, set TP at 2.3%

2. Position Sizing:
   - Reduce position size when entering near resistance
   - If current price is near predicted resistance, use smaller positions

3. Entry Filtering:
   - Avoid long entries when probability curve shows strong resistance ahead
   - Example: If P(>1%)=0.8 but P(>2%)=0.3, expect resistance around 1.5%

4. Dynamic Stop-Loss:
   - For shorts, place stops just above resistance (=support for longs)
   - Adjust stops based on probability gradient

5. Multi-Timeframe Confluence:
   - Combine with traditional S/R levels for confirmation
   - ML resistance + historical resistance = strong level
""")

print('\nDone!')
