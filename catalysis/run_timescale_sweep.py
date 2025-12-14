#!/usr/bin/env python3
'''
Time Architecture Parameter Sweep

Tests fundamental time-scale parameters:
- lookahead_minutes: 30min to 24hr (how far ahead to predict)
- feature_lookback_candles: 24 to 576 candles (2hr to 48hr at 5min)
- volatility_lookback_candles: scaled with feature lookback

Explores whether short-term scalping or long-term positioning works better.
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import polars as pl
from datetime import datetime, timedelta


def run_timescale_sweep(n_permutations=50, months_back_data=20):
    '''
    Parameter sweep focused on time architecture.
    '''

    print('=' * 80)
    print('TIME ARCHITECTURE PARAMETER SWEEP')
    print('Testing prediction horizons from 30min to 24hr')
    print('Testing lookback periods from 2hr to 48hr')
    print('=' * 80)

    # Load data
    print(f'\n[1/3] Loading data ({months_back_data} months)...')
    kline_size = 300  # 5min candles
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back_data * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')

    print(f'Fetching: {start_date_str} to {end_date.strftime("%Y-%m-%d")}')
    print(f'Kline size: {kline_size}s (5 minutes)')

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)

    data = historical.data
    print(f'Loaded {len(data):,} candles')

    test_size = int(len(data) * 0.15)
    test_days = (test_size * kline_size) / (60 * 60 * 24)
    print(f'Expected test set: ~{test_days:.1f} days')

    # Initialize UEL
    print(f'\n[2/3] Initializing UEL with tradeable_regressor_timescale...')

    uel = loop.UniversalExperimentLoop(
        data=historical.data,
        single_file_model=loop.sfm.lightgbm.tradeable_regressor_timescale
    )

    params = loop.sfm.lightgbm.tradeable_regressor_timescale.params()
    print(f'\nTime Architecture Parameters:')
    print(f'  - lookahead_minutes: {params["lookahead_minutes"]}')
    print(f'  - feature_lookback_candles: {params["feature_lookback_candles"]}')
    print(f'  - volatility_lookback_candles: {params["volatility_lookback_candles"]}')

    # Run sweep
    print(f'\n[3/3] Running UEL with {n_permutations} permutations...')
    print('This tests different time horizons for prediction and lookback\n')

    uel.run(
        experiment_name='timescale_sweep',
        n_permutations=n_permutations,
        prep_each_round=True,
        random_search=True
    )

    # Analyze results
    print('\n' + '=' * 80)
    print('TIME ARCHITECTURE RESULTS')
    print('=' * 80)

    log = uel.experiment_log

    # Extract metrics
    results = []
    for i in range(len(log)):
        extras = uel.extras[i]
        val_metrics = extras['val_trading_metrics']
        test_metrics = extras['test_trading_metrics']
        time_config = extras['time_config']

        results.append({
            'id': i,
            'lookahead_min': time_config['lookahead_minutes'],
            'lookback_candles': time_config['feature_lookback_candles'],
            'lookback_hours': time_config['feature_lookback_candles'] * 5 / 60,  # At 5min candles
            'vol_lookback': time_config['volatility_lookback_candles'],
            'val_rmse': log['val_rmse'][i],
            'val_return': val_metrics['total_return'],
            'val_sharpe': val_metrics['sharpe_ratio'],
            'val_trades': val_metrics['num_trades'],
            'val_winrate': val_metrics['win_rate'],
            'test_return': test_metrics['total_return'],
            'test_sharpe': test_metrics['sharpe_ratio'],
            'test_trades': test_metrics['num_trades'],
            'test_winrate': test_metrics['win_rate'],
            'sg_window': extras['sg_config']['sg_window'],
        })

    results_df = pl.DataFrame(results)
    full_results = log.join(results_df, on='id', how='left')

    # Save
    output_file = '/Users/beyondsyntax/Loop/catalysis/timescale_results.csv'
    full_results.write_csv(output_file)
    print(f'\nResults saved to: {output_file}')

    # Analysis
    print('\n' + '-' * 80)
    print('TOP 5 BY TEST SHARPE RATIO')
    print('-' * 80)

    valid_test = results_df.filter(pl.col('test_trades') >= 5)
    if len(valid_test) > 0:
        top_test = valid_test.sort('test_sharpe', descending=True).head(5)
        print(top_test.select(['id', 'test_sharpe', 'test_return', 'test_trades', 'test_winrate',
                               'lookahead_min', 'lookback_hours', 'sg_window']))
    else:
        print("No configs with >= 5 test trades")

    print('\n' + '-' * 80)
    print('TOP 5 BY TEST RETURN')
    print('-' * 80)

    if len(valid_test) > 0:
        top_return = valid_test.sort('test_return', descending=True).head(5)
        print(top_return.select(['id', 'test_return', 'test_sharpe', 'test_trades',
                                 'lookahead_min', 'lookback_hours']))
    else:
        print("No valid test results")

    # Analysis by time horizon
    print('\n' + '=' * 80)
    print('PERFORMANCE BY PREDICTION HORIZON')
    print('=' * 80)

    if len(valid_test) > 0:
        horizon_stats = (results_df
                        .filter(pl.col('test_trades') >= 5)
                        .group_by('lookahead_min')
                        .agg([
                            pl.col('test_return').mean().alias('avg_return'),
                            pl.col('test_sharpe').mean().alias('avg_sharpe'),
                            pl.len().alias('count')
                        ])
                        .sort('lookahead_min'))

        print(horizon_stats)

    print('\n' + '=' * 80)
    print('PERFORMANCE BY LOOKBACK PERIOD')
    print('=' * 80)

    if len(valid_test) > 0:
        # Group by lookback hours (rounded)
        lookback_stats = (results_df
                         .filter(pl.col('test_trades') >= 5)
                         .with_columns((pl.col('lookback_hours') // 4 * 4).alias('lookback_group'))
                         .group_by('lookback_group')
                         .agg([
                             pl.col('test_return').mean().alias('avg_return'),
                             pl.col('test_sharpe').mean().alias('avg_sharpe'),
                             pl.len().alias('count')
                         ])
                         .sort('lookback_group'))

        print(lookback_stats)

    # Overall stats
    print('\n' + '=' * 80)
    print('OVERALL STATISTICS')
    print('=' * 80)

    with_trades = results_df.filter(pl.col('val_trades') > 0)
    profitable = with_trades.filter(pl.col('test_return') > 0)

    print(f"Total configurations tested: {len(results_df)}")
    print(f"Configs with validation trades: {len(with_trades)}")
    print(f"Configs with positive test return: {len(profitable)}")
    print(f"Configs with >= 5 test trades: {len(valid_test)}")

    if len(with_trades) > 0:
        print(f"\nTest set performance (configs with trades):")
        print(f"  Avg Return: {with_trades['test_return'].mean()*100:.2f}%")
        print(f"  Avg Sharpe: {with_trades['test_sharpe'].mean():.3f}")
        print(f"  Avg Win Rate: {with_trades['test_winrate'].mean()*100:.1f}%")

    # Best config details
    if len(valid_test) > 0:
        print('\n' + '=' * 80)
        print('BEST CONFIGURATION DETAILS')
        print('=' * 80)

        best_id = top_test['id'][0]
        best = full_results.filter(pl.col('id') == best_id)

        print(f"\nConfiguration ID: {best_id}")
        print(f"\nTime Architecture:")
        print(f"  Lookahead: {best['lookahead_min'][0]} minutes ({best['lookahead_min'][0]/60:.1f} hours)")
        print(f"  Feature Lookback: {best['lookback_candles'][0]} candles ({best['lookback_hours'][0]:.1f} hours)")
        print(f"  Volatility Lookback: {best['vol_lookback'][0]} candles")

        print(f"\nModel:")
        print(f"  num_leaves: {best['num_leaves'][0]}")
        print(f"  learning_rate: {best['learning_rate'][0]}")
        print(f"  num_iterations: {best['num_iterations'][0]}")
        print(f"  val_rmse: {best['val_rmse'][0]:.6f}")

        print(f"\nTest Performance:")
        print(f"  Sharpe: {best['test_sharpe'][0]:.3f}")
        print(f"  Return: {best['test_return'][0]*100:.2f}%")
        print(f"  Win Rate: {best['test_winrate'][0]*100:.1f}%")
        print(f"  Num Trades: {best['test_trades'][0]}")

    return uel, full_results


if __name__ == '__main__':
    uel, results = run_timescale_sweep(n_permutations=50, months_back_data=20)
    print('\nDone!')
