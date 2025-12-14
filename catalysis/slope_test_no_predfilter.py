#!/usr/bin/env python3
'''
Test bare arithmetic slope WITHOUT min_pred_level filter.

Only use:
- min_slope_change: 0.0001
- min_spacing: 15
- skip_boundary: 10

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


def detect_slope_changes(predictions, min_slope_change=0.0001, skip_boundary=10, min_spacing=15):
    """Detect slope changes - NO min_pred_level filter."""
    slopes = np.diff(predictions)
    crossings = []
    last_bar = -999

    for i in range(2, len(slopes)):
        bar_idx = i + 1
        if bar_idx < skip_boundary or bar_idx >= len(predictions) - skip_boundary:
            continue

        prev_slope = slopes[i-1]
        curr_slope = slopes[i]

        if prev_slope < 0 and curr_slope > 0:
            if abs(prev_slope) > min_slope_change:
                if bar_idx - last_bar >= min_spacing:
                    crossings.append({'bar': bar_idx, 'type': 'bullish'})
                    last_bar = bar_idx

        elif prev_slope > 0 and curr_slope < 0:
            if abs(prev_slope) > min_slope_change:
                if bar_idx - last_bar >= min_spacing:
                    crossings.append({'bar': bar_idx, 'type': 'bearish'})
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
print('BARE ARITHMETIC SLOPE - NO PREDICTION MAGNITUDE FILTER')
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

# Test on same 5 random 6-month periods
candles_per_6mo = int((6 * 30 * 24 * 60) / 5)
max_start = len(full_data) - candles_per_6mo

np.random.seed(42)
start_indices = np.random.randint(0, max_start, 5)

print(f'\nTesting 5 random 6-month periods (NO min_pred_level filter)...')
results = []

for period_idx, start_idx in enumerate(start_indices):
    end_idx = start_idx + candles_per_6mo
    period_data = full_data[start_idx:end_idx]

    print(f'\n[Period {period_idx + 1}/5] Candles {start_idx} to {end_idx} ({len(period_data):,})')

    prep_result = tradeable_regressor.prep(period_data)
    test_data = prep_result['_test_clean']
    predictions = lgb_model.predict(test_data.select(numeric_features).to_numpy())

    # Detect slope changes - NO min_pred_level
    crossings = detect_slope_changes(predictions, min_slope_change=0.0001, min_spacing=15)

    metrics = simulate_trading(test_data, crossings)

    print(f'  Return: {metrics["total_return"]*100:.2f}%')
    print(f'  Sharpe: {metrics["sharpe_ratio"]:.3f}')
    print(f'  Trades: {metrics["num_trades"]} (Win Rate: {metrics["win_rate"]*100:.1f}%)')

    results.append({
        'period': period_idx + 1,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'return': metrics['total_return'],
        'sharpe': metrics['sharpe_ratio'],
        'num_trades': metrics['num_trades'],
        'win_rate': metrics['win_rate'],
    })

results_df = pl.DataFrame(results)

print('\n' + '=' * 80)
print('SUMMARY - NO PREDICTION FILTER')
print('=' * 80)
print(results_df)

profitable = results_df.filter(pl.col('return') > 0)
print(f"\nProfitable periods: {len(profitable)}/5 ({len(profitable)/5*100:.0f}%)")
print(f"Avg Return: {results_df['return'].mean()*100:.2f}%")
print(f"Avg Sharpe: {results_df['sharpe'].mean():.3f}")
print(f"Avg Trades: {results_df['num_trades'].mean():.1f}")
print(f"Avg Win Rate: {results_df['win_rate'].mean()*100:.1f}%")

# Compare to original (with filter)
print('\n' + '=' * 80)
print('COMPARISON: WITH vs WITHOUT min_pred_level filter')
print('=' * 80)

original_results = pl.read_csv('/Users/beyondsyntax/Loop/catalysis/slope_random_periods.csv')

print('\nWITH min_pred_level=0.003:')
print(f"  Avg Return: {original_results['return'].mean()*100:.2f}%")
print(f"  Avg Sharpe: {original_results['sharpe'].mean():.3f}")
print(f"  Avg Trades: {original_results['num_trades'].mean():.1f}")
print(f"  Profitable: {len(original_results.filter(pl.col('return') > 0))}/5")

print('\nWITHOUT min_pred_level filter:')
print(f"  Avg Return: {results_df['return'].mean()*100:.2f}%")
print(f"  Avg Sharpe: {results_df['sharpe'].mean():.3f}")
print(f"  Avg Trades: {results_df['num_trades'].mean():.1f}")
print(f"  Profitable: {len(profitable)}/5")

print('\nDone!')
