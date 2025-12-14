#!/usr/bin/env python3
"""
Test Safer Directional Signals

Compares regular directional signals with "safer" versions:
- Regular LONG: P(LONG | movement)
- Safer LONG: P(LONG and not SHORT | movement) = 1 - P(SHORT | movement)
- Regular SHORT: P(SHORT | movement)
- Safer SHORT: P(SHORT and not LONG | movement) = 1 - P(LONG | movement)
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
print('SAFER DIRECTIONAL SIGNALS TEST')
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
    experiment_name='safer_directional_test',
    n_permutations=1,
    random_search=False
)

# Extract results
print('\n' + '=' * 80)
print('RESULTS')
print('=' * 80)

extras = uel.extras[0]
probs = extras['probabilities']

# Get test period from alignment
alignment = uel._alignment[0]
first_test_dt = alignment['first_test_datetime']
last_test_dt = alignment['last_test_datetime']
print(f'\nTest Period: {first_test_dt} to {last_test_dt}')

# Extract probabilities
p_long = probs['long']
p_short = probs['short']
p_movement = probs['movement']
p_long_given_movement = probs['long_given_movement']
p_short_given_movement = probs['short_given_movement']

# Compute SAFER probabilities
# P(LONG and not SHORT | movement) = 1 - P(SHORT | movement)
p_safe_long = 1.0 - p_short_given_movement
# P(SHORT and not LONG | movement) = 1 - P(LONG | movement)
p_safe_short = 1.0 - p_long_given_movement

print('\n' + '=' * 80)
print('PROBABILITY STATISTICS')
print('=' * 80)

print('\nRegular Probabilities:')
print(f'  P(LONG | movement):   mean={np.mean(p_long_given_movement):.3f}, std={np.std(p_long_given_movement):.3f}')
print(f'  P(SHORT | movement):  mean={np.mean(p_short_given_movement):.3f}, std={np.std(p_short_given_movement):.3f}')
print(f'  P(MOVEMENT):          mean={np.mean(p_movement):.3f}, std={np.std(p_movement):.3f}')

print('\nSafer Probabilities (no opposite direction):')
print(f'  P(LONG and not SHORT | movement):  mean={np.mean(p_safe_long):.3f}, std={np.std(p_safe_long):.3f}')
print(f'  P(SHORT and not LONG | movement):  mean={np.mean(p_safe_short):.3f}, std={np.std(p_safe_short):.3f}')

# Compare signal counts
print('\n' + '=' * 80)
print('SIGNAL COMPARISON (threshold > 0.7, movement > 0.3)')
print('=' * 80)

# Regular signals
regular_long = np.sum((p_long_given_movement > 0.7) & (p_movement > 0.3))
regular_short = np.sum((p_short_given_movement > 0.7) & (p_movement > 0.3))

# Safer signals
safer_long = np.sum((p_safe_long > 0.7) & (p_movement > 0.3))
safer_short = np.sum((p_safe_short > 0.7) & (p_movement > 0.3))

print('\nLONG Signals:')
print(f'  Regular (P(LONG|movement) > 0.7):          {regular_long:5d} ({regular_long/len(p_long)*100:.1f}%)')
print(f'  Safer (1 - P(SHORT|movement) > 0.7):       {safer_long:5d} ({safer_long/len(p_long)*100:.1f}%)')
print(f'  Difference:                                  {safer_long - regular_long:+5d}')

print('\nSHORT Signals:')
print(f'  Regular (P(SHORT|movement) > 0.7):         {regular_short:5d} ({regular_short/len(p_short)*100:.1f}%)')
print(f'  Safer (1 - P(LONG|movement) > 0.7):        {safer_short:5d} ({safer_short/len(p_short)*100:.1f}%)')
print(f'  Difference:                                  {safer_short - regular_short:+5d}')

# Analyze the difference
print('\n' + '=' * 80)
print('SIGNAL OVERLAP ANALYSIS')
print('=' * 80)

# How many signals are in both regular and safer?
long_both = np.sum((p_long_given_movement > 0.7) & (p_safe_long > 0.7) & (p_movement > 0.3))
long_only_regular = np.sum((p_long_given_movement > 0.7) & (p_safe_long <= 0.7) & (p_movement > 0.3))
long_only_safer = np.sum((p_long_given_movement <= 0.7) & (p_safe_long > 0.7) & (p_movement > 0.3))

short_both = np.sum((p_short_given_movement > 0.7) & (p_safe_short > 0.7) & (p_movement > 0.3))
short_only_regular = np.sum((p_short_given_movement > 0.7) & (p_safe_short <= 0.7) & (p_movement > 0.3))
short_only_safer = np.sum((p_short_given_movement <= 0.7) & (p_safe_short > 0.7) & (p_movement > 0.3))

print('\nLONG Signal Overlap:')
print(f'  Both regular and safer:      {long_both:5d}')
print(f'  Only regular:                {long_only_regular:5d} (high SHORT risk)')
print(f'  Only safer:                  {long_only_safer:5d} (low SHORT, but lower LONG)')

print('\nSHORT Signal Overlap:')
print(f'  Both regular and safer:      {short_both:5d}')
print(f'  Only regular:                {short_only_regular:5d} (high LONG risk)')
print(f'  Only safer:                  {short_only_safer:5d} (low LONG, but lower SHORT)')

# Show examples
print('\n' + '=' * 80)
print('EXAMPLE SIGNALS')
print('=' * 80)

# Find examples of each type
print('\nLONG - Only in REGULAR (has SHORT risk):')
idx = np.where((p_long_given_movement > 0.7) & (p_safe_long <= 0.7) & (p_movement > 0.3))[0]
if len(idx) > 0:
    for i in idx[:3]:
        print(f'  Sample {i}: P(LONG|mov)={p_long_given_movement[i]:.3f}, P(SHORT|mov)={p_short_given_movement[i]:.3f}, P(safe_long)={p_safe_long[i]:.3f}')
else:
    print('  None found')

print('\nLONG - Only in SAFER (low SHORT risk, lower LONG):')
idx = np.where((p_long_given_movement <= 0.7) & (p_safe_long > 0.7) & (p_movement > 0.3))[0]
if len(idx) > 0:
    for i in idx[:3]:
        print(f'  Sample {i}: P(LONG|mov)={p_long_given_movement[i]:.3f}, P(SHORT|mov)={p_short_given_movement[i]:.3f}, P(safe_long)={p_safe_long[i]:.3f}')
else:
    print('  None found')

print('\nSHORT - Only in REGULAR (has LONG risk):')
idx = np.where((p_short_given_movement > 0.7) & (p_safe_short <= 0.7) & (p_movement > 0.3))[0]
if len(idx) > 0:
    for i in idx[:3]:
        print(f'  Sample {i}: P(SHORT|mov)={p_short_given_movement[i]:.3f}, P(LONG|mov)={p_long_given_movement[i]:.3f}, P(safe_short)={p_safe_short[i]:.3f}')
else:
    print('  None found')

print('\nSHORT - Only in SAFER (low LONG risk, lower SHORT):')
idx = np.where((p_short_given_movement <= 0.7) & (p_safe_short > 0.7) & (p_movement > 0.3))[0]
if len(idx) > 0:
    for i in idx[:3]:
        print(f'  Sample {i}: P(SHORT|mov)={p_short_given_movement[i]:.3f}, P(LONG|mov)={p_long_given_movement[i]:.3f}, P(safe_short)={p_safe_short[i]:.3f}')
else:
    print('  None found')

# Correlation analysis
print('\n' + '=' * 80)
print('CORRELATION ANALYSIS')
print('=' * 80)

corr_long_safe = np.corrcoef(p_long_given_movement, p_safe_long)[0, 1]
corr_short_safe = np.corrcoef(p_short_given_movement, p_safe_short)[0, 1]

print(f'\nCorrelation between P(LONG|mov) and P(safe_long):   {corr_long_safe:.3f}')
print(f'Correlation between P(SHORT|mov) and P(safe_short): {corr_short_safe:.3f}')

# Save results
print('\n' + '=' * 80)
print('SAVING RESULTS')
print('=' * 80)

# Create results DataFrame
results_df = pl.DataFrame({
    'p_long_given_movement': p_long_given_movement,
    'p_short_given_movement': p_short_given_movement,
    'p_movement': p_movement,
    'p_safe_long': p_safe_long,
    'p_safe_short': p_safe_short,
    'p_long': p_long,
    'p_short': p_short
})

output_file = '/Users/beyondsyntax/Loop/catalysis/safer_directional_results.csv'
results_df.write_csv(output_file)
print(f'\nResults saved to: {output_file}')

print('\n' + '=' * 80)
print('Done!')
