#!/usr/bin/env python3
'''
Test Slope-Based Trading on All Regressor SFMs

Tests the slope inflection trading logic on:
1. tradeable_regressor (baseline - already tested)
2. tradeable_regressor_sg (with SG filter in prep)
3. tradeable_regressor_sg_1stderiv (with SG + 1st derivative)
4. tradeable_regressor_timescale (with time architecture params)

For each SFM:
- Train model on same data
- Apply slope-based detection
- Test best configs from original sweep
- Save results for comparison
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import numpy as np
import polars as pl
import pickle
from datetime import datetime, timedelta
from pathlib import Path


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
print('SLOPE-BASED TRADING - ALL REGRESSOR SFMS COMPARISON')
print('=' * 80)

# Load data once
print('\nLoading 20 months of data...')
kline_size = 300
end_date = datetime.now()
start_date = end_date - timedelta(days=20 * 30)
start_date_str = start_date.strftime('%Y-%m-%d')

historical = loop.HistoricalData()
historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
full_data = historical.data
print(f'Loaded {len(full_data):,} candles')

# Setup test periods
candles_per_6mo = int((6 * 30 * 24 * 60) / 5)
max_start = len(full_data) - candles_per_6mo

np.random.seed(42)
start_indices = np.random.randint(0, max_start, 5)

# Test configurations (top 5 from original sweep)
test_configs = [
    {'name': 'Best Overall', 'slope_delta': 0.0005, 'sustained': 6, 'spacing': 8},
    {'name': 'Best Sharpe', 'slope_delta': 0.0006, 'sustained': 6, 'spacing': 18},
    {'name': 'Config 3', 'slope_delta': 0.00015, 'sustained': 5, 'spacing': 20},
    {'name': 'Config 4', 'slope_delta': 0.0005, 'sustained': 6, 'spacing': 18},
    {'name': 'Config 5', 'slope_delta': 0.0004, 'sustained': 6, 'spacing': 35},
]

# SFMs to test
sfms = [
    {
        'name': 'tradeable_regressor',
        'module': loop.sfm.lightgbm.tradeable_regressor,
        'config': {
            'lookahead_minutes': 30,
            'feature_lookback_period': 96,
            'volatility_lookback_candles': 144,
            'volatility_lookback': 96,
        }
    },
    {
        'name': 'tradeable_regressor_sg',
        'module': loop.sfm.lightgbm.tradeable_regressor_sg,
        'config': {}  # Will use defaults
    },
    {
        'name': 'tradeable_regressor_sg_1stderiv',
        'module': loop.sfm.lightgbm.tradeable_regressor_sg_1stderiv,
        'config': {}  # Will use defaults
    },
    {
        'name': 'tradeable_regressor_timescale',
        'module': loop.sfm.lightgbm.tradeable_regressor_timescale,
        'config': {
            'lookahead_minutes': 30,
            'feature_lookback_candles': 96,
        }
    },
]

all_sfm_results = []

# Model params
model_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_iterations': 100,
    'force_col_wise': True,
}

# Test each SFM
for sfm_idx, sfm_info in enumerate(sfms):
    print(f'\n{"=" * 80}')
    print(f'SFM {sfm_idx + 1}/{len(sfms)}: {sfm_info["name"]}')
    print('=' * 80)

    sfm = sfm_info['module']
    config = sfm_info['config']

    # Patch config if needed
    if config and hasattr(sfm, 'CONFIG'):
        for key, value in config.items():
            sfm.CONFIG[key] = value

    # Train model on full data
    print('\n[1/3] Training model...')
    try:
        prep_result = sfm.prep(full_data)
        model_result = sfm.model(prep_result, model_params)

        lgb_model = model_result['models'][0]
        numeric_features = model_result['extras']['numeric_features']

        print(f'Model trained with {len(numeric_features)} features')

        # Prepare test periods
        print('\n[2/3] Preparing test periods...')
        period_predictions = []
        period_test_data = []

        for period_idx, start_idx in enumerate(start_indices):
            end_idx = start_idx + candles_per_6mo
            period_data = full_data[start_idx:end_idx]

            period_prep = sfm.prep(period_data)
            test_data = period_prep['_test_clean']
            predictions = lgb_model.predict(test_data.select(numeric_features).to_numpy())

            period_predictions.append(predictions)
            period_test_data.append(test_data)

        print(f'Prepared {len(period_predictions)} periods')

        # Test each config
        print('\n[3/3] Testing slope configs...')
        config_results = []

        for config_idx, slope_config in enumerate(test_configs):
            period_metrics = []

            for period_idx in range(5):
                predictions = period_predictions[period_idx]
                test_data = period_test_data[period_idx]

                crossings = detect_slope_changes_proper(
                    predictions,
                    min_slope_delta=slope_config['slope_delta'],
                    min_sustained_bars=slope_config['sustained'],
                    min_spacing=slope_config['spacing']
                )

                metrics = simulate_trading(test_data, crossings)
                period_metrics.append(metrics)

            # Aggregate
            avg_return = np.mean([m['total_return'] for m in period_metrics])
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in period_metrics])
            avg_trades = np.mean([m['num_trades'] for m in period_metrics])
            avg_winrate = np.mean([m['win_rate'] for m in period_metrics])
            profitable_periods = sum([1 for m in period_metrics if m['total_return'] > 0])

            config_results.append({
                'sfm': sfm_info['name'],
                'config_name': slope_config['name'],
                'slope_delta': slope_config['slope_delta'],
                'sustained': slope_config['sustained'],
                'spacing': slope_config['spacing'],
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'avg_trades': avg_trades,
                'avg_winrate': avg_winrate,
                'profitable_periods': profitable_periods,
            })

        # Show results for this SFM
        results_df = pl.DataFrame(config_results)
        print('\nResults:')
        print(results_df.select(['config_name', 'avg_return', 'avg_sharpe', 'avg_trades', 'profitable_periods']))

        all_sfm_results.extend(config_results)

    except Exception as e:
        print(f'\nERROR with {sfm_info["name"]}: {e}')
        print('Skipping this SFM...')
        continue

# Final comparison
print('\n' + '=' * 80)
print('FINAL COMPARISON ACROSS ALL SFMS')
print('=' * 80)

all_results_df = pl.DataFrame(all_sfm_results)

# Save
output_file = '/Users/beyondsyntax/Loop/catalysis/all_sfm_comparison.csv'
all_results_df.write_csv(output_file)
print(f'\nFull results saved to: {output_file}')

# Best by SFM
print('\n' + '-' * 80)
print('BEST CONFIG FOR EACH SFM')
print('-' * 80)

for sfm_name in all_results_df['sfm'].unique():
    sfm_results = all_results_df.filter(pl.col('sfm') == sfm_name)
    best = sfm_results.sort('avg_return', descending=True).head(1)

    print(f'\n{sfm_name}:')
    print(f'  Config: {best["config_name"][0]}')
    print(f'  Avg Return: {best["avg_return"][0]*100:.2f}%')
    print(f'  Avg Sharpe: {best["avg_sharpe"][0]:.3f}')
    print(f'  Avg Trades: {best["avg_trades"][0]:.1f}')
    print(f'  Profitable: {best["profitable_periods"][0]}/5')

# Overall best
print('\n' + '-' * 80)
print('TOP 5 OVERALL ACROSS ALL SFMS')
print('-' * 80)

top_overall = all_results_df.sort('avg_return', descending=True).head(5)
print(top_overall.select(['sfm', 'config_name', 'avg_return', 'avg_sharpe', 'profitable_periods']))

print('\n' + '-' * 80)
print('SUMMARY STATISTICS BY SFM')
print('-' * 80)

summary = (all_results_df
          .group_by('sfm')
          .agg([
              pl.col('avg_return').mean().alias('mean_return'),
              pl.col('avg_return').max().alias('max_return'),
              pl.col('avg_sharpe').mean().alias('mean_sharpe'),
              pl.col('avg_trades').mean().alias('mean_trades'),
              (pl.col('avg_return') > 0).sum().alias('profitable_configs'),
              pl.len().alias('total_configs'),
          ]))

print(summary)

print('\nDone!')
