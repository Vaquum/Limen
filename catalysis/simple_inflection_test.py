#!/usr/bin/env python3
'''
Simple Inflection Trading Logic Test

Tests two simple approaches on best time architecture config:
- 30min lookahead
- 8hr (96 candle) lookback
- 144 volatility lookback

Strategy 1: EMA Crossover
Strategy 2: Bare Arithmetic (slope changes)
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import numpy as np
import polars as pl
from datetime import datetime, timedelta


def ema(data, window):
    """Calculate exponential moving average."""
    alpha = 2 / (window + 1)
    ema_values = np.zeros_like(data)
    ema_values[0] = data[0]

    for i in range(1, len(data)):
        ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]

    return ema_values


def detect_ema_crossovers(predictions, fast_window=5, slow_window=15, min_pred_level=0.003, skip_boundary=10, min_spacing=10):
    """Detect EMA crossovers on predictions."""
    fast_ema = ema(predictions, fast_window)
    slow_ema = ema(predictions, slow_window)

    crossings = []
    last_bar = -999

    for i in range(1, len(predictions)):
        if i < skip_boundary or i >= len(predictions) - skip_boundary:
            continue

        if predictions[i] < min_pred_level:
            continue

        # Fast crosses above slow = bullish
        if fast_ema[i-1] <= slow_ema[i-1] and fast_ema[i] > slow_ema[i]:
            if i - last_bar >= min_spacing:
                crossings.append({'bar': i, 'type': 'bullish'})
                last_bar = i

        # Fast crosses below slow = bearish
        elif fast_ema[i-1] >= slow_ema[i-1] and fast_ema[i] < slow_ema[i]:
            if i - last_bar >= min_spacing:
                crossings.append({'bar': i, 'type': 'bearish'})
                last_bar = i

    return pl.DataFrame(crossings) if crossings else pl.DataFrame()


def detect_slope_changes(predictions, min_slope_change=0.00001, min_pred_level=0.003, skip_boundary=10, min_spacing=10):
    """Detect slope changes using bare arithmetic."""
    slopes = np.diff(predictions)

    crossings = []
    last_bar = -999

    for i in range(2, len(slopes)):
        bar_idx = i + 1  # Account for diff offset

        if bar_idx < skip_boundary or bar_idx >= len(predictions) - skip_boundary:
            continue

        if predictions[bar_idx] < min_pred_level:
            continue

        prev_slope = slopes[i-1]
        curr_slope = slopes[i]

        # Slope changes from negative to positive = bullish inflection
        if prev_slope < 0 and curr_slope > 0:
            if abs(prev_slope) > min_slope_change:
                if bar_idx - last_bar >= min_spacing:
                    crossings.append({'bar': bar_idx, 'type': 'bullish'})
                    last_bar = bar_idx

        # Slope changes from positive to negative = bearish inflection
        elif prev_slope > 0 and curr_slope < 0:
            if abs(prev_slope) > min_slope_change:
                if bar_idx - last_bar >= min_spacing:
                    crossings.append({'bar': bar_idx, 'type': 'bearish'})
                    last_bar = bar_idx

    return pl.DataFrame(crossings) if crossings else pl.DataFrame()


def simulate_trading(data, crossings, commission_rate=0.002):
    """Simulate LONG-only trading on crossings."""
    if len(crossings) == 0:
        return {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade': 0,
            'sharpe_ratio': 0,
        }

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
            trades.append({'net_return': net_return, 'bars_held': bar - position['entry_bar']})
            position = None

    # Close final position
    if position is not None:
        bar = len(prices) - 1
        exit_price = prices[bar]
        gross_return = (exit_price - position['entry_price']) / position['entry_price']
        net_return = gross_return - commission_rate
        trades.append({'net_return': net_return, 'bars_held': bar - position['entry_bar']})

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
        'avg_bars_held': trades_df['bars_held'].mean(),
    }


def test_simple_strategies():
    print('=' * 80)
    print('SIMPLE INFLECTION TRADING TEST')
    print('Using best config: 30min lookahead, 8hr lookback')
    print('=' * 80)

    # Load data
    print('\n[1/4] Loading data (20 months)...')
    kline_size = 300
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
    data = historical.data
    print(f'Loaded {len(data):,} candles')

    # Prepare data with best config params
    print('\n[2/4] Preparing data with best time architecture...')

    CONFIG = {
        'lookahead_minutes': 30,
        'feature_lookback_period': 96,  # 8 hours
        'volatility_lookback_candles': 144,
        'volatility_lookback': 96,
    }

    # Import and patch config
    from loop.sfm.lightgbm import tradeable_regressor
    for key, value in CONFIG.items():
        tradeable_regressor.CONFIG[key] = value

    prep_result = tradeable_regressor.prep(data)

    # Train simple model
    print('\n[3/4] Training model...')

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

    model_result = tradeable_regressor.model(prep_result, model_params)

    # Get predictions
    lgb_model = model_result['models'][0]
    numeric_features = model_result['extras']['numeric_features']

    val_clean = prep_result['_val_clean']
    test_clean = model_result['extras']['test_clean']

    val_predictions = lgb_model.predict(val_clean.select(numeric_features).to_numpy())
    test_predictions = model_result['_preds']

    print(f'Val predictions: {len(val_predictions)}')
    print(f'Test predictions: {len(test_predictions)}')

    # Test strategies
    print('\n[4/4] Testing simple strategies...')

    print('\n' + '=' * 80)
    print('STRATEGY 1: EMA CROSSOVER (Fast=5, Slow=15)')
    print('=' * 80)

    # Test different EMA params
    ema_configs = [
        {'fast': 3, 'slow': 10, 'spacing': 5},
        {'fast': 5, 'slow': 15, 'spacing': 10},
        {'fast': 7, 'slow': 21, 'spacing': 15},
    ]

    ema_results = []
    for config in ema_configs:
        val_crossings = detect_ema_crossovers(
            val_predictions,
            fast_window=config['fast'],
            slow_window=config['slow'],
            min_spacing=config['spacing']
        )

        test_crossings = detect_ema_crossovers(
            test_predictions,
            fast_window=config['fast'],
            slow_window=config['slow'],
            min_spacing=config['spacing']
        )

        val_metrics = simulate_trading(val_clean, val_crossings)
        test_metrics = simulate_trading(test_clean, test_crossings)

        ema_results.append({
            'fast': config['fast'],
            'slow': config['slow'],
            'spacing': config['spacing'],
            'val_return': val_metrics['total_return'],
            'val_sharpe': val_metrics['sharpe_ratio'],
            'val_trades': val_metrics['num_trades'],
            'val_winrate': val_metrics['win_rate'],
            'test_return': test_metrics['total_return'],
            'test_sharpe': test_metrics['sharpe_ratio'],
            'test_trades': test_metrics['num_trades'],
            'test_winrate': test_metrics['win_rate'],
        })

    ema_df = pl.DataFrame(ema_results)
    print('\nEMA Crossover Results:')
    print(ema_df)

    print('\n' + '=' * 80)
    print('STRATEGY 2: BARE ARITHMETIC (Slope Changes)')
    print('=' * 80)

    # Test different slope thresholds
    slope_configs = [
        {'min_slope': 0.00001, 'spacing': 5},
        {'min_slope': 0.00005, 'spacing': 10},
        {'min_slope': 0.0001, 'spacing': 15},
    ]

    slope_results = []
    for config in slope_configs:
        val_crossings = detect_slope_changes(
            val_predictions,
            min_slope_change=config['min_slope'],
            min_spacing=config['spacing']
        )

        test_crossings = detect_slope_changes(
            test_predictions,
            min_slope_change=config['min_slope'],
            min_spacing=config['spacing']
        )

        val_metrics = simulate_trading(val_clean, val_crossings)
        test_metrics = simulate_trading(test_clean, test_crossings)

        slope_results.append({
            'min_slope': config['min_slope'],
            'spacing': config['spacing'],
            'val_return': val_metrics['total_return'],
            'val_sharpe': val_metrics['sharpe_ratio'],
            'val_trades': val_metrics['num_trades'],
            'val_winrate': val_metrics['win_rate'],
            'test_return': test_metrics['total_return'],
            'test_sharpe': test_metrics['sharpe_ratio'],
            'test_trades': test_metrics['num_trades'],
            'test_winrate': test_metrics['win_rate'],
        })

    slope_df = pl.DataFrame(slope_results)
    print('\nSlope Change Results:')
    print(slope_df)

    # Best performers
    print('\n' + '=' * 80)
    print('BEST PERFORMERS')
    print('=' * 80)

    best_ema = ema_df.sort('test_sharpe', descending=True).head(1)
    best_slope = slope_df.sort('test_sharpe', descending=True).head(1)

    print('\nBest EMA Config:')
    print(f"  Fast={best_ema['fast'][0]}, Slow={best_ema['slow'][0]}, Spacing={best_ema['spacing'][0]}")
    print(f"  Test Sharpe: {best_ema['test_sharpe'][0]:.3f}")
    print(f"  Test Return: {best_ema['test_return'][0]*100:.2f}%")
    print(f"  Test Trades: {best_ema['test_trades'][0]} (Win Rate: {best_ema['test_winrate'][0]*100:.1f}%)")

    print('\nBest Slope Config:')
    print(f"  Min Slope={best_slope['min_slope'][0]:.6f}, Spacing={best_slope['spacing'][0]}")
    print(f"  Test Sharpe: {best_slope['test_sharpe'][0]:.3f}")
    print(f"  Test Return: {best_slope['test_return'][0]*100:.2f}%")
    print(f"  Test Trades: {best_slope['test_trades'][0]} (Win Rate: {best_slope['test_winrate'][0]*100:.1f}%)")


if __name__ == '__main__':
    test_simple_strategies()
    print('\nDone!')
