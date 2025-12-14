#!/usr/bin/env python3
'''
Comprehensive Parameter Sweep for Slope-Based Trading

Sweeps:
- min_slope_delta: 0.00005 to 0.0008 (inflection strength)
- min_sustained_bars: 2 to 7 (persistence requirement)
- min_spacing: 8 to 35 (signal spacing)

Tests on 5 random 6-month periods, saves all results.
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import numpy as np
import polars as pl
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product


def detect_slope_changes_proper(predictions, min_slope_delta=0.0002, min_sustained_bars=3, skip_boundary=10, min_spacing=15):
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
print('SLOPE-BASED TRADING - COMPREHENSIVE PARAMETER SWEEP')
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

# Define parameter grid
slope_deltas = [0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00040, 0.00050, 0.00060, 0.00080]
sustained_bars = [2, 3, 4, 5, 6, 7]
spacings = [8, 10, 12, 15, 18, 20, 25, 30, 35]

# Random sample from full grid
np.random.seed(42)
all_combinations = list(product(slope_deltas, sustained_bars, spacings))
n_permutations = min(100, len(all_combinations))  # Test 100 random combinations
sampled_combinations = [all_combinations[i] for i in np.random.choice(len(all_combinations), n_permutations, replace=False)]

print(f'\nParameter space: {len(all_combinations)} total combinations')
print(f'Testing {n_permutations} random configurations')
print(f'  slope_delta range: {min(slope_deltas):.5f} to {max(slope_deltas):.5f}')
print(f'  sustained_bars range: {min(sustained_bars)} to {max(sustained_bars)}')
print(f'  spacing range: {min(spacings)} to {max(spacings)}')

# Setup test periods
candles_per_6mo = int((6 * 30 * 24 * 60) / 5)
max_start = len(full_data) - candles_per_6mo

np.random.seed(42)
start_indices = np.random.randint(0, max_start, 5)

# Prepare all period data
print('\nPreparing 5 test periods...')
period_predictions = []
period_test_data = []

for period_idx, start_idx in enumerate(start_indices):
    end_idx = start_idx + candles_per_6mo
    period_data = full_data[start_idx:end_idx]

    prep_result = tradeable_regressor.prep(period_data)
    test_data = prep_result['_test_clean']
    predictions = lgb_model.predict(test_data.select(numeric_features).to_numpy())

    period_predictions.append(predictions)
    period_test_data.append(test_data)
    print(f'  Period {period_idx + 1}: {len(predictions):,} predictions')

# Run sweep
print(f'\nRunning sweep ({n_permutations} configs × 5 periods = {n_permutations * 5} tests)...')
all_results = []

for config_idx, (slope_delta, sustained, spacing) in enumerate(sampled_combinations):
    if (config_idx + 1) % 10 == 0:
        print(f'  Progress: {config_idx + 1}/{n_permutations} configs tested...')

    period_results = []

    for period_idx in range(5):
        predictions = period_predictions[period_idx]
        test_data = period_test_data[period_idx]

        crossings = detect_slope_changes_proper(
            predictions,
            min_slope_delta=slope_delta,
            min_sustained_bars=sustained,
            min_spacing=spacing
        )

        metrics = simulate_trading(test_data, crossings)
        period_results.append(metrics)

    # Aggregate metrics across periods
    avg_return = np.mean([m['total_return'] for m in period_results])
    avg_sharpe = np.mean([m['sharpe_ratio'] for m in period_results])
    avg_trades = np.mean([m['num_trades'] for m in period_results])
    avg_winrate = np.mean([m['win_rate'] for m in period_results])
    profitable_periods = sum([1 for m in period_results if m['total_return'] > 0])

    all_results.append({
        'config_id': config_idx,
        'slope_delta': slope_delta,
        'sustained_bars': sustained,
        'spacing': spacing,
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'avg_trades': avg_trades,
        'avg_winrate': avg_winrate,
        'profitable_periods': profitable_periods,
        'p1_return': period_results[0]['total_return'],
        'p2_return': period_results[1]['total_return'],
        'p3_return': period_results[2]['total_return'],
        'p4_return': period_results[3]['total_return'],
        'p5_return': period_results[4]['total_return'],
        'p1_trades': period_results[0]['num_trades'],
        'p2_trades': period_results[1]['num_trades'],
        'p3_trades': period_results[2]['num_trades'],
        'p4_trades': period_results[3]['num_trades'],
        'p5_trades': period_results[4]['num_trades'],
    })

print('\nSweep complete!')

# Convert to DataFrame
results_df = pl.DataFrame(all_results)

# Save
output_file = '/Users/beyondsyntax/Loop/catalysis/slope_param_sweep_results.csv'
results_df.write_csv(output_file)
print(f'Results saved to: {output_file}')

# Analysis
print('\n' + '=' * 80)
print('TOP 10 BY AVERAGE RETURN')
print('=' * 80)
top_return = results_df.sort('avg_return', descending=True).head(10)
print(top_return.select(['config_id', 'slope_delta', 'sustained_bars', 'spacing',
                         'avg_return', 'avg_sharpe', 'avg_trades', 'profitable_periods']))

print('\n' + '=' * 80)
print('TOP 10 BY SHARPE RATIO')
print('=' * 80)
top_sharpe = results_df.sort('avg_sharpe', descending=True).head(10)
print(top_sharpe.select(['config_id', 'slope_delta', 'sustained_bars', 'spacing',
                        'avg_return', 'avg_sharpe', 'avg_trades', 'profitable_periods']))

print('\n' + '=' * 80)
print('CONFIGS WITH 5/5 PROFITABLE PERIODS')
print('=' * 80)
perfect_configs = results_df.filter(pl.col('profitable_periods') == 5)
if len(perfect_configs) > 0:
    print(perfect_configs.sort('avg_return', descending=True).select([
        'config_id', 'slope_delta', 'sustained_bars', 'spacing',
        'avg_return', 'avg_sharpe', 'avg_trades'
    ]))
else:
    print('No configs with 5/5 profitable periods')

print('\n' + '=' * 80)
print('CONFIGS WITH 4/5 PROFITABLE PERIODS')
print('=' * 80)
good_configs = results_df.filter(pl.col('profitable_periods') == 4).sort('avg_return', descending=True).head(10)
if len(good_configs) > 0:
    print(good_configs.select([
        'config_id', 'slope_delta', 'sustained_bars', 'spacing',
        'avg_return', 'avg_sharpe', 'avg_trades'
    ]))
else:
    print('No configs with 4/5 profitable periods')

# Overall statistics
print('\n' + '=' * 80)
print('OVERALL STATISTICS')
print('=' * 80)
profitable = results_df.filter(pl.col('avg_return') > 0)
print(f'Configs tested: {len(results_df)}')
print(f'Profitable configs (avg_return > 0): {len(profitable)} ({len(profitable)/len(results_df)*100:.1f}%)')
print(f'Mean avg_return: {results_df["avg_return"].mean()*100:.2f}%')
print(f'Median avg_return: {results_df["avg_return"].median()*100:.2f}%')
print(f'Best avg_return: {results_df["avg_return"].max()*100:.2f}%')
print(f'Worst avg_return: {results_df["avg_return"].min()*100:.2f}%')

print('\n' + '=' * 80)
print('BEST CONFIGURATION DETAILS')
print('=' * 80)
best_idx = results_df['avg_return'].arg_max()
best = results_df[best_idx]

print(f'\nConfig ID: {best["config_id"][0]}')
print(f'Parameters:')
print(f'  slope_delta: {best["slope_delta"][0]:.5f}')
print(f'  sustained_bars: {best["sustained_bars"][0]}')
print(f'  spacing: {best["spacing"][0]}')
print(f'\nPerformance:')
print(f'  Avg Return: {best["avg_return"][0]*100:.2f}%')
print(f'  Avg Sharpe: {best["avg_sharpe"][0]:.3f}')
print(f'  Avg Trades: {best["avg_trades"][0]:.1f}')
print(f'  Avg Win Rate: {best["avg_winrate"][0]*100:.1f}%')
print(f'  Profitable Periods: {best["profitable_periods"][0]}/5')
print(f'\nPer-Period Returns:')
print(f'  Period 1: {best["p1_return"][0]*100:.2f}% ({best["p1_trades"][0]} trades)')
print(f'  Period 2: {best["p2_return"][0]*100:.2f}% ({best["p2_trades"][0]} trades)')
print(f'  Period 3: {best["p3_return"][0]*100:.2f}% ({best["p3_trades"][0]} trades)')
print(f'  Period 4: {best["p4_return"][0]*100:.2f}% ({best["p4_trades"][0]} trades)')
print(f'  Period 5: {best["p5_return"][0]*100:.2f}% ({best["p5_trades"][0]} trades)')

print('\nDone!')
