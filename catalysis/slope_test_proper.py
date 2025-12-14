#!/usr/bin/env python3
'''
Proper Slope-Based Trading - Filter on SLOPE characteristics only.

Filters:
1. min_slope_delta: Minimum change in slope (inflection strength)
2. min_sustained_bars: New slope direction must persist for N bars
3. min_spacing: Minimum bars between signals
4. skip_boundary: Skip edges

NO prediction magnitude filtering.
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import numpy as np
import polars as pl
import pickle
from datetime import datetime, timedelta
from pathlib import Path


def detect_slope_changes_proper(predictions, min_slope_delta=0.0002, min_sustained_bars=3, skip_boundary=10, min_spacing=15):
    """
    Detect slope changes using SLOPE characteristics only.

    Args:
        min_slope_delta: Minimum abs(curr_slope - prev_slope) for valid inflection
        min_sustained_bars: New slope direction must persist for this many bars
        skip_boundary: Skip first/last N bars
        min_spacing: Minimum bars between signals
    """
    slopes = np.diff(predictions)
    crossings = []
    last_bar = -999

    for i in range(2, len(slopes) - min_sustained_bars):
        bar_idx = i + 1

        if bar_idx < skip_boundary or bar_idx >= len(predictions) - skip_boundary:
            continue

        prev_slope = slopes[i-1]
        curr_slope = slopes[i]

        # Check if slope direction changed
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

        # Filter 1: Check slope change magnitude (inflection strength)
        slope_delta = abs(curr_slope - prev_slope)
        if slope_delta < min_slope_delta:
            continue

        # Filter 2: Verify sustained direction
        # For bullish: next N slopes should be mostly positive
        # For bearish: next N slopes should be mostly negative
        future_slopes = slopes[i:i+min_sustained_bars]

        if signal_type == 'bullish':
            sustained = np.sum(future_slopes > 0) >= min_sustained_bars * 0.67  # 2/3 positive
        else:  # bearish
            sustained = np.sum(future_slopes < 0) >= min_sustained_bars * 0.67  # 2/3 negative

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
print('PROPER SLOPE-BASED TRADING')
print('Filters on slope characteristics ONLY (no prediction magnitude)')
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

# Test different slope-based configs
slope_configs = [
    {'min_slope_delta': 0.0001, 'sustained': 2, 'spacing': 10},
    {'min_slope_delta': 0.0002, 'sustained': 3, 'spacing': 15},
    {'min_slope_delta': 0.0003, 'sustained': 4, 'spacing': 20},
    {'min_slope_delta': 0.0005, 'sustained': 5, 'spacing': 25},
]

candles_per_6mo = int((6 * 30 * 24 * 60) / 5)
max_start = len(full_data) - candles_per_6mo

np.random.seed(42)
start_indices = np.random.randint(0, max_start, 5)

all_configs_results = []

for config_idx, slope_config in enumerate(slope_configs):
    print(f'\n{"=" * 80}')
    print(f'CONFIG {config_idx + 1}: slope_delta={slope_config["min_slope_delta"]}, sustained={slope_config["sustained"]}, spacing={slope_config["spacing"]}')
    print('=' * 80)

    results = []

    for period_idx, start_idx in enumerate(start_indices):
        end_idx = start_idx + candles_per_6mo
        period_data = full_data[start_idx:end_idx]

        prep_result = tradeable_regressor.prep(period_data)
        test_data = prep_result['_test_clean']
        predictions = lgb_model.predict(test_data.select(numeric_features).to_numpy())

        crossings = detect_slope_changes_proper(
            predictions,
            min_slope_delta=slope_config['min_slope_delta'],
            min_sustained_bars=slope_config['sustained'],
            min_spacing=slope_config['spacing']
        )

        metrics = simulate_trading(test_data, crossings)

        results.append({
            'period': period_idx + 1,
            'return': metrics['total_return'],
            'sharpe': metrics['sharpe_ratio'],
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
        })

    results_df = pl.DataFrame(results)

    profitable = results_df.filter(pl.col('return') > 0)
    avg_return = results_df['return'].mean()
    avg_sharpe = results_df['sharpe'].mean()
    avg_trades = results_df['num_trades'].mean()

    print(f"\nProfitable periods: {len(profitable)}/5 ({len(profitable)/5*100:.0f}%)")
    print(f"Avg Return: {avg_return*100:.2f}%")
    print(f"Avg Sharpe: {avg_sharpe:.3f}")
    print(f"Avg Trades: {avg_trades:.1f}")

    all_configs_results.append({
        'config': f"delta={slope_config['min_slope_delta']},sust={slope_config['sustained']},sp={slope_config['spacing']}",
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'avg_trades': avg_trades,
        'profitable_pct': len(profitable) / 5,
    })

# Summary
print('\n' + '=' * 80)
print('SUMMARY ACROSS CONFIGS')
print('=' * 80)

summary_df = pl.DataFrame(all_configs_results)
print(summary_df)

best_idx = summary_df['avg_return'].arg_max()
print(f'\nBest config: {summary_df["config"][best_idx]}')
print(f'  Avg Return: {summary_df["avg_return"][best_idx]*100:.2f}%')
print(f'  Avg Sharpe: {summary_df["avg_sharpe"][best_idx]:.3f}')

print('\nDone!')
