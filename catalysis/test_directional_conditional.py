#!/usr/bin/env python3
"""
Test Directional Conditional Probability Model

Trains 3 models (LONG, SHORT, MOVEMENT) and computes conditional probabilities.
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

print('=' * 80)
print('DIRECTIONAL CONDITIONAL PROBABILITY MODEL TEST')
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

# Train with UEL
print('\nTraining directional conditional models...')

uel = loop.UniversalExperimentLoop(
    data=full_data,
    single_file_model=dc
)

uel.run(
    experiment_name='directional_conditional_test',
    n_permutations=1,
    random_search=False
)

# Extract results
print('\n' + '=' * 80)
print('RESULTS')
print('=' * 80)

log = uel.experiment_log
extras = uel.extras[0]

# Get test period from alignment
alignment = uel._alignment[0]
first_test_dt = alignment['first_test_datetime']
last_test_dt = alignment['last_test_datetime']
print(f'\nTest Period: {first_test_dt} to {last_test_dt}')

# Model performance
print('\nLONG Model Performance:')
print(f'  Accuracy:  {log["accuracy"][0]:.4f}')
print(f'  Precision: {log["precision"][0]:.4f}')
print(f'  Recall:    {log["recall"][0]:.4f}')
if 'roc_auc' in log.columns:
    print(f'  ROC AUC:   {log["roc_auc"][0]:.4f}')

# Class distributions
print('\nClass Distributions:')
for task in ['long', 'short', 'movement']:
    dist = extras['class_distributions'][task]
    print(f'  {task.upper():8s}: {dist}')

# Probability statistics
probs = extras['probabilities']

print('\nProbability Statistics:')
print(f'  P(LONG):              mean={np.mean(probs["long"]):.3f}, std={np.std(probs["long"]):.3f}')
print(f'  P(SHORT):             mean={np.mean(probs["short"]):.3f}, std={np.std(probs["short"]):.3f}')
print(f'  P(MOVEMENT):          mean={np.mean(probs["movement"]):.3f}, std={np.std(probs["movement"]):.3f}')
print(f'  P(LONG | movement):   mean={np.mean(probs["long_given_movement"]):.3f}, std={np.std(probs["long_given_movement"]):.3f}')
print(f'  P(SHORT | movement):  mean={np.mean(probs["short_given_movement"]):.3f}, std={np.std(probs["short_given_movement"]):.3f}')

# Analyze conditional probabilities
print('\n' + '=' * 80)
print('CONDITIONAL PROBABILITY ANALYSIS')
print('=' * 80)

p_long = probs['long']
p_short = probs['short']
p_movement = probs['movement']
p_long_given_movement = probs['long_given_movement']
p_short_given_movement = probs['short_given_movement']

# Find examples of strong directional signals
strong_long_indices = np.where((p_long_given_movement > 0.7) & (p_movement > 0.3))[0]
strong_short_indices = np.where((p_short_given_movement > 0.7) & (p_movement > 0.3))[0]

print(f'\nStrong LONG signals (P(LONG|movement)>0.7 & P(movement)>0.3): {len(strong_long_indices)} samples')
print(f'Strong SHORT signals (P(SHORT|movement)>0.7 & P(movement)>0.3): {len(strong_short_indices)} samples')

if len(strong_long_indices) > 0:
    print('\nExample STRONG LONG signals:')
    for i in strong_long_indices[:5]:
        print(f'  Sample {i}: P(LONG)={p_long[i]:.3f}, P(movement)={p_movement[i]:.3f}, P(LONG|movement)={p_long_given_movement[i]:.3f}')

if len(strong_short_indices) > 0:
    print('\nExample STRONG SHORT signals:')
    for i in strong_short_indices[:5]:
        print(f'  Sample {i}: P(SHORT)={p_short[i]:.3f}, P(movement)={p_movement[i]:.3f}, P(SHORT|movement)={p_short_given_movement[i]:.3f}')

# Analyze low movement samples
low_movement_indices = np.where(p_movement < 0.2)[0]
print(f'\nLow movement samples (P(movement)<0.2): {len(low_movement_indices)} samples')
if len(low_movement_indices) > 0:
    print('  These samples have low volatility predictions - conditional probabilities may be unreliable')

# Compare raw vs conditional probabilities
print('\n' + '=' * 80)
print('RAW vs CONDITIONAL COMPARISON')
print('=' * 80)

# Correlation between raw and conditional
corr_long = np.corrcoef(p_long, p_long_given_movement)[0, 1]
corr_short = np.corrcoef(p_short, p_short_given_movement)[0, 1]

print(f'\nCorrelation between P(LONG) and P(LONG|movement):   {corr_long:.3f}')
print(f'Correlation between P(SHORT) and P(SHORT|movement): {corr_short:.3f}')

# Find samples where conditioning changes prediction significantly
long_diff = np.abs(p_long - p_long_given_movement)
short_diff = np.abs(p_short - p_short_given_movement)

big_change_indices = np.where(long_diff > 0.3)[0]
print(f'\nSamples where conditioning changes LONG probability by >0.3: {len(big_change_indices)}')

if len(big_change_indices) > 0:
    print('\nExamples of big changes:')
    for i in big_change_indices[:5]:
        print(f'  Sample {i}: P(LONG)={p_long[i]:.3f} -> P(LONG|movement)={p_long_given_movement[i]:.3f}, P(movement)={p_movement[i]:.3f}')

print('\n' + '=' * 80)
print('Done!')
