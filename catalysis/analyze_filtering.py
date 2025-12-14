#!/usr/bin/env python3
'''
Analyze how filtering reduces raw slope changes to final signals.

Shows the cascade:
1. Raw prediction curve slope changes
2. After min_slope_change filter
3. After min_pred_level filter
4. After boundary filter
5. After spacing filter
6. Final signals
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import numpy as np
import polars as pl
import pickle
from datetime import datetime, timedelta
from pathlib import Path


print('=' * 80)
print('FILTERING ANALYSIS - BARE ARITHMETIC SLOPE DETECTION')
print('=' * 80)

# Load pickled model
pickle_path = Path('/Users/beyondsyntax/Loop/catalysis/slope_model.pkl')
with open(pickle_path, 'rb') as f:
    model_data = pickle.load(f)

lgb_model = model_data['lgb_model']
numeric_features = model_data['numeric_features']
config = model_data['config']

print(f'\nLoaded model with {len(numeric_features)} features')

# Patch config
from loop.sfm.lightgbm import tradeable_regressor
for key, value in config.items():
    tradeable_regressor.CONFIG[key] = value

# Load data - use first 6 months from random test
print('\nLoading 20 months of data...')
kline_size = 300
end_date = datetime.now()
start_date = end_date - timedelta(days=20 * 30)
start_date_str = start_date.strftime('%Y-%m-%d')

historical = loop.HistoricalData()
historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
full_data = historical.data

# Use same random period as test (first one from seed=42)
candles_per_6mo = int((6 * 30 * 24 * 60) / 5)
np.random.seed(42)
start_idx = np.random.randint(0, len(full_data) - candles_per_6mo, 1)[0]
period_data = full_data[start_idx:start_idx + candles_per_6mo]

print(f'Analyzing period: candles {start_idx} to {start_idx + candles_per_6mo}')
print(f'Total candles in period: {len(period_data):,}')

# Prepare and predict
prep_result = tradeable_regressor.prep(period_data)
test_data = prep_result['_test_clean']
predictions = lgb_model.predict(test_data.select(numeric_features).to_numpy())

print(f'\nPredictions generated: {len(predictions):,}')

# Analyze filtering cascade
slopes = np.diff(predictions)

print('\n' + '=' * 80)
print('FILTERING CASCADE')
print('=' * 80)

# Step 1: Raw slope changes
raw_changes = 0
for i in range(2, len(slopes)):
    prev_slope = slopes[i-1]
    curr_slope = slopes[i]
    if (prev_slope < 0 and curr_slope > 0) or (prev_slope > 0 and curr_slope < 0):
        raw_changes += 1

print(f'\n[1] Raw slope sign changes: {raw_changes:,}')
print(f'    (any prev_slope vs curr_slope sign change)')

# Step 2: After min_slope_change
min_slope_change = 0.0001
after_min_slope = 0
for i in range(2, len(slopes)):
    prev_slope = slopes[i-1]
    curr_slope = slopes[i]
    if (prev_slope < 0 and curr_slope > 0) or (prev_slope > 0 and curr_slope < 0):
        if abs(prev_slope) > min_slope_change:
            after_min_slope += 1

print(f'\n[2] After min_slope_change filter (>{min_slope_change}): {after_min_slope:,}')
print(f'    Removed: {raw_changes - after_min_slope:,} ({(raw_changes - after_min_slope)/raw_changes*100:.1f}%)')

# Step 3: After boundary filter
skip_boundary = 10
after_boundary = 0
for i in range(2, len(slopes)):
    bar_idx = i + 1
    if bar_idx < skip_boundary or bar_idx >= len(predictions) - skip_boundary:
        continue
    prev_slope = slopes[i-1]
    curr_slope = slopes[i]
    if (prev_slope < 0 and curr_slope > 0) or (prev_slope > 0 and curr_slope < 0):
        if abs(prev_slope) > min_slope_change:
            after_boundary += 1

print(f'\n[3] After boundary filter (skip first/last {skip_boundary}): {after_boundary:,}')
print(f'    Removed: {after_min_slope - after_boundary:,} ({(after_min_slope - after_boundary)/after_min_slope*100:.1f}%)')

# Step 4: After min_pred_level
min_pred_level = 0.003
after_pred_level = 0
for i in range(2, len(slopes)):
    bar_idx = i + 1
    if bar_idx < skip_boundary or bar_idx >= len(predictions) - skip_boundary:
        continue
    if predictions[bar_idx] < min_pred_level:
        continue
    prev_slope = slopes[i-1]
    curr_slope = slopes[i]
    if (prev_slope < 0 and curr_slope > 0) or (prev_slope > 0 and curr_slope < 0):
        if abs(prev_slope) > min_slope_change:
            after_pred_level += 1

print(f'\n[4] After min_pred_level filter (>={min_pred_level}): {after_pred_level:,}')
print(f'    Removed: {after_boundary - after_pred_level:,} ({(after_boundary - after_pred_level)/after_boundary*100:.1f}%)')

# Step 5: After spacing filter
min_spacing = 15
final_signals = 0
last_bar = -999
for i in range(2, len(slopes)):
    bar_idx = i + 1
    if bar_idx < skip_boundary or bar_idx >= len(predictions) - skip_boundary:
        continue
    if predictions[bar_idx] < min_pred_level:
        continue
    prev_slope = slopes[i-1]
    curr_slope = slopes[i]
    if (prev_slope < 0 and curr_slope > 0) or (prev_slope > 0 and curr_slope < 0):
        if abs(prev_slope) > min_slope_change:
            if bar_idx - last_bar >= min_spacing:
                final_signals += 1
                last_bar = bar_idx

print(f'\n[5] After spacing filter (>={min_spacing} candles): {final_signals:,}')
print(f'    Removed: {after_pred_level - final_signals:,} ({(after_pred_level - final_signals)/after_pred_level*100:.1f}%)')

# Summary
print('\n' + '=' * 80)
print('SUMMARY')
print('=' * 80)
print(f'Total prediction candles: {len(predictions):,}')
print(f'Raw slope changes: {raw_changes:,} ({raw_changes/len(predictions)*100:.2f}% of candles)')
print(f'Final signals: {final_signals:,} ({final_signals/len(predictions)*100:.3f}% of candles)')
print(f'Overall reduction: {(1 - final_signals/raw_changes)*100:.1f}%')
print(f'\nAverage candles between signals: {len(predictions)/final_signals:.0f}')
print(f'Average hours between signals: {(len(predictions)/final_signals * 5 / 60):.1f}')

# Prediction statistics
print('\n' + '=' * 80)
print('PREDICTION STATISTICS')
print('=' * 80)
print(f'Mean prediction: {predictions.mean():.6f}')
print(f'Std prediction: {predictions.std():.6f}')
print(f'Min prediction: {predictions.min():.6f}')
print(f'Max prediction: {predictions.max():.6f}')
print(f'Predictions >= {min_pred_level}: {(predictions >= min_pred_level).sum():,} ({(predictions >= min_pred_level).sum()/len(predictions)*100:.1f}%)')

# Slope statistics
print('\n' + '=' * 80)
print('SLOPE STATISTICS')
print('=' * 80)
print(f'Mean |slope|: {np.abs(slopes).mean():.6f}')
print(f'Std slope: {slopes.std():.6f}')
print(f'Min slope: {slopes.min():.6f}')
print(f'Max slope: {slopes.max():.6f}')
print(f'Slopes > {min_slope_change}: {(np.abs(slopes) > min_slope_change).sum():,} ({(np.abs(slopes) > min_slope_change).sum()/len(slopes)*100:.1f}%)')

print('\nDone!')
