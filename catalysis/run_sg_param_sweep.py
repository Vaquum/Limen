#!/usr/bin/env python3
'''
SG Filter Parameter Sweep using UEL

Runs comprehensive parameter sweep over both model and SG filter parameters
to find optimal configurations for inflection-based trading.

This uses UEL's built-in random search to explore the parameter space efficiently.
'''

import warnings
warnings.filterwarnings('ignore')

import loop
import polars as pl
from datetime import datetime, timedelta


def run_sg_param_sweep(n_permutations=100, months_back_data=20):
    '''
    Execute parameter sweep for SG filter trading strategy.

    Args:
        n_permutations: Number of random parameter combinations to try
        months_back_data: Months of historical data to fetch

    Returns:
        UEL object with all results
    '''

    print('=' * 80)
    print('SG Filter Trading - Parameter Sweep via UEL')
    print('=' * 80)

    # Load data
    print(f'\n[1/3] Loading data ({months_back_data} months)...')
    kline_size = 300
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back_data * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')

    print(f'Fetching: {start_date_str} to {end_date.strftime("%Y-%m-%d")}')

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=kline_size, start_date_limit=start_date_str)

    data = historical.data
    print(f'Loaded {len(data):,} candles')

    # Expected test period
    test_size = int(len(data) * 0.15)
    test_days = (test_size * kline_size) / (60 * 60 * 24)
    print(f'Expected test set: ~{test_days:.1f} days')

    # Initialize UEL with SG filter SFM
    print(f'\n[2/3] Initializing UEL with tradeable_regressor_sg...')

    uel = loop.UniversalExperimentLoop(
        data=historical.data,
        single_file_model=loop.sfm.lightgbm.tradeable_regressor_sg
    )

    print(f'Parameter space includes:')
    params = loop.sfm.lightgbm.tradeable_regressor_sg.params()
    print(f'  - Model params: {len([k for k in params.keys() if not k.startswith("sg_") and not k.startswith("min_") and k != "lookahead_minutes"])}')
    print(f'  - Lookahead: lookahead_minutes (4 options: 30, 60, 90, 120 min)')
    print(f'  - SG params: sg_window (5 options), sg_polyorder (3 options)')
    print(f'  - Filter params: min_curvature (5), min_prediction (5), min_spacing (4)')

    # Run parameter sweep
    print(f'\n[3/3] Running UEL with {n_permutations} permutations...')
    print('This will take a while - each permutation trains a model and simulates trading.')
    print('Progress will be shown below:\n')

    uel.run(
        experiment_name='sg_filter_param_sweep',
        n_permutations=n_permutations,
        prep_each_round=True,
        random_search=True
    )

    # Analyze results
    print('\n' + '=' * 80)
    print('PARAMETER SWEEP RESULTS')
    print('=' * 80)

    # Get experiment log
    log = uel.experiment_log

    # Add trading metrics to log from extras
    trading_metrics = []
    for i in range(len(log)):
        extras = uel.extras[i]
        val_metrics = extras['val_trading_metrics']
        test_metrics = extras['test_trading_metrics']

        # Get lookahead from log
        lookahead = log['lookahead_minutes'][i] if 'lookahead_minutes' in log.columns else 90

        trading_metrics.append({
            'id': i,
            'val_total_return': val_metrics['total_return'],
            'val_num_trades': val_metrics['num_trades'],
            'val_win_rate': val_metrics['win_rate'],
            'val_avg_trade': val_metrics['avg_trade'],
            'val_sharpe': val_metrics['sharpe_ratio'],
            'test_total_return': test_metrics['total_return'],
            'test_num_trades': test_metrics['num_trades'],
            'test_win_rate': test_metrics['win_rate'],
            'test_avg_trade': test_metrics['avg_trade'],
            'test_sharpe': test_metrics['sharpe_ratio'],
            'lookahead_minutes': lookahead,
            'sg_window': extras['sg_config']['sg_window'],
            'sg_polyorder': extras['sg_config']['sg_polyorder'],
            'min_curvature': extras['sg_config']['min_curvature_threshold'],
            'min_prediction': extras['sg_config']['min_prediction_level'],
            'min_spacing': extras['sg_config']['min_inflection_spacing'],
        })

    trading_df = pl.DataFrame(trading_metrics)

    # Merge with experiment log
    full_results = log.join(trading_df, on='id', how='left')

    # Save full results
    output_file = '/Users/beyondsyntax/Loop/catalysis/sg_param_sweep_results.csv'
    full_results.write_csv(output_file)
    print(f'\nFull results saved to: {output_file}')

    # Display top performers by different metrics
    print('\n' + '-' * 80)
    print('TOP 5 BY VALIDATION SHARPE RATIO (Primary Optimization Target)')
    print('-' * 80)

    # Filter for configs with at least 10 trades
    valid_results = full_results.filter(pl.col('val_num_trades') >= 10)

    if len(valid_results) > 0:
        top_sharpe = valid_results.sort('val_sharpe', descending=True).head(5)
        print(top_sharpe.select(['id', 'val_sharpe', 'val_total_return', 'val_win_rate',
                                'val_num_trades', 'lookahead_minutes', 'sg_window', 'sg_polyorder',
                                'min_curvature', 'min_prediction']))

        print('\n' + '-' * 80)
        print('TOP 5 BY VALIDATION TOTAL RETURN')
        print('-' * 80)
        top_return = valid_results.sort('val_total_return', descending=True).head(5)
        print(top_return.select(['id', 'val_total_return', 'val_sharpe', 'val_win_rate',
                                'val_num_trades', 'lookahead_minutes', 'sg_window', 'sg_polyorder',
                                'min_curvature', 'min_prediction']))

        print('\n' + '-' * 80)
        print('TOP 5 BY VALIDATION WIN RATE')
        print('-' * 80)
        top_winrate = valid_results.sort('val_win_rate', descending=True).head(5)
        print(top_winrate.select(['id', 'val_win_rate', 'val_total_return', 'val_sharpe',
                                 'val_num_trades', 'lookahead_minutes', 'sg_window', 'sg_polyorder',
                                 'min_curvature', 'min_prediction']))

        # Show test set performance of best validation Sharpe config
        print('\n' + '=' * 80)
        print('BEST VALIDATION CONFIG - TEST SET PERFORMANCE')
        print('=' * 80)

        best_id = top_sharpe['id'][0]
        best_row = full_results.filter(pl.col('id') == best_id)

        print(f"\nConfiguration ID: {best_id}")
        print(f"Lookahead Minutes: {best_row['lookahead_minutes'][0]}")
        print(f"SG Window: {best_row['sg_window'][0]}")
        print(f"SG Polyorder: {best_row['sg_polyorder'][0]}")
        print(f"Min Curvature: {best_row['min_curvature'][0]:.6f}")
        print(f"Min Prediction: {best_row['min_prediction'][0]:.6f}")
        print(f"Min Spacing: {best_row['min_spacing'][0]}")

        print(f"\nValidation Performance:")
        print(f"  Sharpe: {best_row['val_sharpe'][0]:.3f}")
        print(f"  Total Return: {best_row['val_total_return'][0]*100:.2f}%")
        print(f"  Win Rate: {best_row['val_win_rate'][0]*100:.1f}%")
        print(f"  Avg Trade: {best_row['val_avg_trade'][0]*100:.3f}%")
        print(f"  Num Trades: {best_row['val_num_trades'][0]}")

        print(f"\nTest Set Performance:")
        print(f"  Sharpe: {best_row['test_sharpe'][0]:.3f}")
        print(f"  Total Return: {best_row['test_total_return'][0]*100:.2f}%")
        print(f"  Win Rate: {best_row['test_win_rate'][0]*100:.1f}%")
        print(f"  Avg Trade: {best_row['test_avg_trade'][0]*100:.3f}%")
        print(f"  Num Trades: {best_row['test_num_trades'][0]}")

    else:
        print("\nNO VALID RESULTS: All configurations produced < 10 trades on validation set")
        print("This suggests filters are too strict or SG parameters are not detecting inflections")

    # Overall statistics
    print('\n' + '=' * 80)
    print('OVERALL STATISTICS')
    print('=' * 80)

    configs_with_trades = full_results.filter(pl.col('val_num_trades') > 0)
    configs_profitable = configs_with_trades.filter(pl.col('val_total_return') > 0)

    print(f"Total configurations tested: {len(full_results)}")
    print(f"Configs with at least 1 trade: {len(configs_with_trades)}")
    print(f"Configs with positive return: {len(configs_profitable)}")
    print(f"Configs with >10 trades: {len(valid_results)}")

    if len(configs_with_trades) > 0:
        print(f"\nValidation set statistics (all configs with trades):")
        print(f"  Avg Sharpe: {configs_with_trades['val_sharpe'].mean():.3f}")
        print(f"  Avg Return: {configs_with_trades['val_total_return'].mean()*100:.2f}%")
        print(f"  Avg Win Rate: {configs_with_trades['val_win_rate'].mean()*100:.1f}%")
        print(f"  Avg Num Trades: {configs_with_trades['val_num_trades'].mean():.1f}")

    return uel, full_results


if __name__ == '__main__':
    # Run with 100 permutations and 20 months of data
    # Adjust n_permutations higher (200-500) for more thorough search
    uel, results = run_sg_param_sweep(n_permutations=100, months_back_data=20)
    print('\nDone!')
