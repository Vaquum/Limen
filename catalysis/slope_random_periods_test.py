#!/usr/bin/env python3
'''
Bare Arithmetic Slope Trading - Random Period Tests

Trains model once, pickles it, then tests on random 6-month periods.
Uses best config from simple_inflection_test:
- 30min lookahead
- 8hr lookback
- min_slope=0.0001, spacing=15
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import numpy as np
import polars as pl
import pickle
from datetime import datetime, timedelta
from pathlib import Path


def detect_slope_changes(predictions, min_slope_change=0.0001, min_pred_level=0.003, skip_boundary=10, min_spacing=15):
    """Detect slope changes using bare arithmetic."""
    slopes = np.diff(predictions)

    crossings = []
    last_bar = -999

    for i in range(2, len(slopes)):
        bar_idx = i + 1

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


def train_and_pickle_model():
    """Train model with best config and pickle it."""
    print('=' * 80)
    print('TRAINING AND PICKLING MODEL')
    print('=' * 80)

    # Load full data
    print('\nLoading data (20 months)...')
    kline_size = 300
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)
    full_data = historical.data
    print(f'Loaded {len(full_data):,} candles')

    # Best config from timescale sweep
    CONFIG = {
        'lookahead_minutes': 30,
        'feature_lookback_period': 96,  # 8 hours
        'volatility_lookback_candles': 144,
        'volatility_lookback': 96,
    }

    # Patch config
    from loop.sfm.lightgbm import tradeable_regressor
    for key, value in CONFIG.items():
        tradeable_regressor.CONFIG[key] = value

    print('\nPreparing data...')
    prep_result = tradeable_regressor.prep(full_data)

    print('Training model...')
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

    # Extract model and features
    lgb_model = model_result['models'][0]
    numeric_features = model_result['extras']['numeric_features']

    # Pickle
    pickle_path = Path('/Users/beyondsyntax/Loop/catalysis/slope_model.pkl')
    model_data = {
        'lgb_model': lgb_model,
        'numeric_features': numeric_features,
        'config': CONFIG,
    }

    with open(pickle_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f'\nModel pickled to: {pickle_path}')
    print(f'Features: {len(numeric_features)}')

    return full_data, CONFIG


def test_random_periods(full_data, config, n_periods=5):
    """Test model on random 6-month periods."""
    print('\n' + '=' * 80)
    print(f'TESTING ON {n_periods} RANDOM 6-MONTH PERIODS')
    print('=' * 80)

    # Load pickled model
    pickle_path = Path('/Users/beyondsyntax/Loop/catalysis/slope_model.pkl')
    with open(pickle_path, 'rb') as f:
        model_data = pickle.load(f)

    lgb_model = model_data['lgb_model']
    numeric_features = model_data['numeric_features']

    # Calculate candles per 6 months (at 5min candles)
    candles_per_6mo = int((6 * 30 * 24 * 60) / 5)  # ~52,000 candles
    min_start = 0
    max_start = len(full_data) - candles_per_6mo

    if max_start <= 0:
        print("Not enough data for 6-month periods")
        return

    print(f'\nFull dataset: {len(full_data):,} candles')
    print(f'6-month period: {candles_per_6mo:,} candles')
    print(f'Possible start positions: 0 to {max_start:,}')

    # Prepare data with config
    from loop.sfm.lightgbm import tradeable_regressor
    for key, value in config.items():
        tradeable_regressor.CONFIG[key] = value

    # Generate random periods
    np.random.seed(42)
    start_indices = np.random.randint(min_start, max_start, n_periods)

    results = []

    for period_idx, start_idx in enumerate(start_indices):
        end_idx = start_idx + candles_per_6mo
        period_data = full_data[start_idx:end_idx]

        print(f'\n[Period {period_idx + 1}/{n_periods}] Candles {start_idx} to {end_idx}')
        print(f'  Candles: {len(period_data):,}')

        # Prepare features for period data (without train/val/test split)
        from loop.sfm.lightgbm.tradeable_regressor import (
            standardize_datetime_column, calculate_returns_if_missing,
            market_regime, calculate_dynamic_parameters, calculate_microstructure_features,
            momentum_confirmation
        )

        period_df = period_data.clone()
        period_df = standardize_datetime_column(period_df)
        period_df = calculate_returns_if_missing(period_df)

        period_df = period_df.with_columns([
            (pl.col('close') / pl.col('close').shift(1)).log().alias('log_returns'),
            pl.col('returns').rolling_std(config['volatility_lookback_candles'], min_samples=1).alias('vol_60h'),
            pl.lit(0.5).alias('vol_percentile'),
            pl.lit('normal').alias('volatility_regime'),
            pl.lit(False).alias('regime_low'),
            pl.lit(True).alias('regime_normal'),
            pl.lit(False).alias('regime_high')
        ])

        # Import full CONFIG
        from loop.sfm.lightgbm.tradeable_regressor import CONFIG as BASE_CONFIG

        period_df = market_regime(period_df, 48)
        period_df = calculate_dynamic_parameters(period_df, BASE_CONFIG)
        period_df = calculate_microstructure_features(period_df, BASE_CONFIG)
        period_df = period_df.with_columns([pl.lit(1.0).alias('momentum_score')])

        # Drop NaNs
        period_clean = period_df.drop_nulls()

        # Generate predictions
        predictions = lgb_model.predict(period_clean.select(numeric_features).to_numpy())

        # Detect slope changes
        crossings = detect_slope_changes(predictions, min_slope_change=0.0001, min_spacing=15)

        # Simulate trading
        metrics = simulate_trading(period_clean, crossings)

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
            'avg_trade': metrics['avg_trade'],
        })

    # Summary
    results_df = pl.DataFrame(results)

    print('\n' + '=' * 80)
    print('SUMMARY OF RANDOM 6-MONTH PERIODS')
    print('=' * 80)
    print(results_df)

    print('\n' + '-' * 80)
    print('AGGREGATE STATISTICS')
    print('-' * 80)

    profitable = results_df.filter(pl.col('return') > 0)
    print(f"Profitable periods: {len(profitable)}/{len(results_df)} ({len(profitable)/len(results_df)*100:.1f}%)")
    print(f"Avg Return: {results_df['return'].mean()*100:.2f}%")
    print(f"Avg Sharpe: {results_df['sharpe'].mean():.3f}")
    print(f"Avg Trades: {results_df['num_trades'].mean():.1f}")
    print(f"Avg Win Rate: {results_df['win_rate'].mean()*100:.1f}%")

    print(f"\nBest Period: {results_df.sort('return', descending=True).head(1)['period'][0]}")
    print(f"  Return: {results_df.sort('return', descending=True).head(1)['return'][0]*100:.2f}%")

    print(f"\nWorst Period: {results_df.sort('return', descending=False).head(1)['period'][0]}")
    print(f"  Return: {results_df.sort('return', descending=False).head(1)['return'][0]*100:.2f}%")

    # Save results
    output_file = '/Users/beyondsyntax/Loop/catalysis/slope_random_periods.csv'
    results_df.write_csv(output_file)
    print(f'\nResults saved to: {output_file}')


if __name__ == '__main__':
    # Train and pickle model
    full_data, config = train_and_pickle_model()

    # Test on random 6-month periods
    test_random_periods(full_data, config, n_periods=5)

    print('\nDone!')
