#!/usr/bin/env python3
'''
Test Slope-Based Trading on Breakout Regressor

The breakout_regressor_ridge predicts breakout percentages (continuous values).
We'll apply the same slope inflection logic to these predictions.

Using a forked version that uses full dataset instead of random_slice.
'''

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')

import loop
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import breakout_regressor_fulldata


def detect_slope_changes_proper(predictions, min_slope_delta=0.0005,
                                min_sustained_bars=6, skip_boundary=10,
                                min_spacing=8):
    """Detect slope changes using SLOPE characteristics only."""
    slopes = np.diff(predictions)
    crossings = []
    last_bar = -999

    for i in range(2, len(slopes) - min_sustained_bars):
        bar_idx = i + 1

        if bar_idx < skip_boundary or bar_idx >= len(predictions) - skip_boundary:
            continue

        prev_slope = slopes[i-1]
        curr_slope = slopes[i]

        direction_changed = False
        signal_type = None

        if prev_slope < 0 and curr_slope > 0:
            direction_changed = True
            signal_type = 'bullish'
        elif prev_slope > 0 and curr_slope < 0:
            direction_changed = True
            signal_type = 'bearish'

        if not direction_changed:
            continue

        # Filter 1: Slope change magnitude
        slope_delta = abs(curr_slope - prev_slope)
        if slope_delta < min_slope_delta:
            continue

        # Filter 2: Sustained direction
        future_slopes = slopes[i:i+min_sustained_bars]

        if signal_type == 'bullish':
            sustained = np.sum(future_slopes > 0) >= min_sustained_bars * 0.67
        else:
            sustained = np.sum(future_slopes < 0) >= min_sustained_bars * 0.67

        if not sustained:
            continue

        # Filter 3: Spacing
        if bar_idx - last_bar < min_spacing:
            continue

        crossings.append({'bar': bar_idx, 'type': signal_type})
        last_bar = bar_idx

    return pl.DataFrame(crossings) if crossings else pl.DataFrame()


def simulate_trading(data, crossings, commission_rate=0.002):
    """Simulate LONG-only trading."""
    if len(crossings) == 0:
        return {'total_return': 0, 'num_trades': 0, 'win_rate': 0, 'avg_trade': 0, 'sharpe_ratio': 0}

    trades = []
    position = None
    prices = data['close'].to_numpy()

    for row in crossings.iter_rows(named=True):
        bar = row['bar']
        crossing_type = row['type']

        if crossing_type == 'bullish' and position is None:
            position = {'entry_bar': bar, 'entry_price': prices[bar]}
        elif crossing_type == 'bearish' and position is not None:
            exit_price = prices[bar]
            gross_return = (exit_price - position['entry_price']) / position['entry_price']
            net_return = gross_return - commission_rate
            trades.append({'net_return': net_return})
            position = None

    if position is not None:
        bar = len(prices) - 1
        exit_price = prices[bar]
        gross_return = (exit_price - position['entry_price']) / position['entry_price']
        net_return = gross_return - commission_rate
        trades.append({'net_return': net_return})

    if len(trades) == 0:
        return {'total_return': 0, 'num_trades': 0, 'win_rate': 0, 'avg_trade': 0, 'sharpe_ratio': 0}

    trades_df = pl.DataFrame(trades)
    wins = trades_df.filter(pl.col('net_return') > 0)
    returns = trades_df['net_return'].to_numpy()
    sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0

    return {
        'total_return': trades_df['net_return'].sum(),
        'num_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
        'avg_trade': trades_df['net_return'].mean(),
        'sharpe_ratio': sharpe,
    }


print('=' * 80)
print('SLOPE-BASED TRADING ON BREAKOUT REGRESSOR')
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

# Use UEL with modified breakout_regressor (full dataset)
print('\nTraining breakout regressor with Manifest (full dataset)...')

# Get the Manifest object from the forked SFM
sfm = breakout_regressor_fulldata
manifest = sfm.manifest()

# Create UEL with the SFM
uel = loop.UniversalExperimentLoop(
    data=full_data,
    single_file_model=sfm
)

# Run with 1 permutation, passing the manifest
uel.run(
    experiment_name='breakout_regressor_slope_test',
    n_permutations=1,
    prep_each_round=True,
    random_search=False,
    manifest=manifest
)

print('\nExtracting predictions...')

# Get the model and data from UEL
log = uel.experiment_log
model_id = 0

# Check what's available in UEL
print(f'\nUEL log columns: {log.columns}')
print(f'\nUEL has {len(uel.extras)} extras')
print(f'UEL has {len(uel.preds)} preds')
print(f'UEL has {len(uel.models)} models')
print(f'UEL has {len(uel._alignment)} alignment')

# For Manifest-based models, predictions might be in uel.preds
if len(uel.preds) > 0:
    predictions = uel.preds[model_id]
    print(f'\nFound {len(predictions)} predictions in uel.preds')
else:
    print('ERROR: No predictions found')
    exit(1)

# Get test data from alignment - use datetime range to extract from full_data
if len(uel._alignment) > 0:
    alignment = uel._alignment[model_id]
    print(f'\nAlignment keys: {alignment.keys()}')

    # Extract test data using datetime range
    first_test_dt = alignment['first_test_datetime']
    last_test_dt = alignment['last_test_datetime']

    print(f'Test period: {first_test_dt} to {last_test_dt}')

    # Filter full_data to get test data
    test_data = full_data.filter(
        (pl.col('datetime') >= first_test_dt) &
        (pl.col('datetime') <= last_test_dt)
    )
else:
    print('ERROR: No alignment data found')
    exit(1)

print(f'Test data: {len(test_data)} rows')
print(f'Test data columns: {test_data.columns}')

# Ensure we have close prices
if 'close' not in test_data.columns:
    print('ERROR: Test data missing close prices')
    exit(1)

# Test slope configs
print('\nTesting slope configurations...')

test_configs = [
    {'name': 'Best Overall', 'slope_delta': 0.0005, 'sustained': 6, 'spacing': 8},
    {'name': 'Best Sharpe', 'slope_delta': 0.0006, 'sustained': 6, 'spacing': 18},
    {'name': 'Config 3', 'slope_delta': 0.00015, 'sustained': 5, 'spacing': 20},
]

results = []

for config in test_configs:
    crossings = detect_slope_changes_proper(
        predictions,
        min_slope_delta=config['slope_delta'],
        min_sustained_bars=config['sustained'],
        min_spacing=config['spacing']
    )

    metrics = simulate_trading(test_data, crossings)

    results.append({
        'config': config['name'],
        'slope_delta': config['slope_delta'],
        'sustained': config['sustained'],
        'spacing': config['spacing'],
        'return': metrics['total_return'],
        'sharpe': metrics['sharpe_ratio'],
        'num_trades': metrics['num_trades'],
        'win_rate': metrics['win_rate'],
    })

    print(f"\n{config['name']}:")
    print(f"  Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
    print(f"  Trades: {metrics['num_trades']} (Win Rate: {metrics['win_rate']*100:.1f}%)")

# Summary
results_df = pl.DataFrame(results)

print('\n' + '=' * 80)
print('RESULTS')
print('=' * 80)
print(results_df)

# Save
output_file = '/Users/beyondsyntax/Loop/catalysis/breakout_regressor_slopes.csv'
results_df.write_csv(output_file)
print(f'\nResults saved to: {output_file}')

print('\nDone!')
