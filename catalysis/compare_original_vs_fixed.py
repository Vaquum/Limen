#!/usr/bin/env python3
'''
Compare original vs fixed tradeline_long_binary SFM
Tests index alignment bug fix using 3 months of 2024 data
Sweeps time-related model parameters only
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import loop.sfm.lightgbm.tradeline_long_binary as sfm_original
import catalysis.tradeline_long_binary_fixed as sfm_fixed
import polars as pl

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
START_DATE = '2024-01-01'
N_ROWS = 2160  # 90 days * 24 hours = 3 months

# Time-related params only for sweep
TIME_PARAMS = {
    'lookahead_hours': [24, 48, 72],
    'max_duration_hours': [24, 48, 72],
    'loser_timeout_hours': [12, 24, 48],
    'max_hold_hours': [24, 48, 72],
}

# Fix other params to reduce permutation count
FIXED_PARAMS = {
    'quantile_threshold': [0.75],
    'min_height_pct': [0.003],
    'long_threshold_percentile': [75],
    'confidence_threshold': [0.50],
    'position_size': [0.20],
    'min_stop_loss': [0.010],
    'max_stop_loss': [0.040],
    'atr_stop_multiplier': [1.5],
    'trailing_activation': [0.02],
    'trailing_distance': [0.5],
    'default_atr_pct': [0.015],
    'num_leaves': [31],
    'learning_rate': [0.05],
    'feature_fraction': [0.9],
    'bagging_fraction': [0.8],
    'bagging_freq': [5],
    'min_child_samples': [20],
    'lambda_l1': [0],
    'lambda_l2': [0],
    'n_estimators': [500],
}


def create_custom_params_original():
    '''Override params() for original SFM with time-focused sweep'''
    return {**TIME_PARAMS, **FIXED_PARAMS}


def create_custom_params_fixed():
    '''Override params() for fixed SFM with time-focused sweep'''
    return {**TIME_PARAMS, **FIXED_PARAMS}


def run_comparison():
    print('=' * 80)
    print('TRADELINE LONG BINARY: ORIGINAL vs FIXED COMPARISON')
    print('=' * 80)
    print(f'Period: {START_DATE}, {N_ROWS} hourly candles (3 months)')
    print(f'Kline size: {KLINE_SIZE}s (1 hour)')
    print(f'Time params being swept:')
    for k, v in TIME_PARAMS.items():
        print(f'  {k}: {v}')
    print('=' * 80)

    # Load data
    print('\nLoading historical data...')
    historical = loop.HistoricalData()
    historical.get_spot_klines(
        kline_size=KLINE_SIZE,
        start_date_limit=START_DATE,
        n_rows=N_ROWS
    )
    data = historical.data

    print(f'Loaded {len(data):,} candles')
    print(f'Date range: {data["datetime"].min()} to {data["datetime"].max()}')

    if len(data) < 500:
        print(f'ERROR: Insufficient data ({len(data)} rows)')
        return None

    # Monkey-patch params functions for time-focused sweep
    sfm_original.params = create_custom_params_original
    sfm_fixed.params = create_custom_params_fixed

    # Calculate expected permutations
    n_perms = 1
    for v in TIME_PARAMS.values():
        n_perms *= len(v)
    print(f'\nExpected permutations per SFM: {n_perms}')

    results = {}

    # Run ORIGINAL SFM
    print('\n' + '=' * 80)
    print('RUNNING ORIGINAL SFM (with index misalignment bug)')
    print('=' * 80)

    try:
        uel_original = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_original
        )

        uel_original.run(
            experiment_name='tradeline_original_compare',
            n_permutations=n_perms,
            prep_each_round=True
        )

        results['original'] = {
            'log': uel_original.experiment_log,
            'status': 'completed'
        }
        print(f'Original SFM completed: {len(uel_original.experiment_log)} permutations')

    except Exception as e:
        print(f'ERROR running original SFM: {e}')
        import traceback
        traceback.print_exc()
        results['original'] = {'status': 'error', 'error': str(e)}

    # Run FIXED SFM
    print('\n' + '=' * 80)
    print('RUNNING FIXED SFM (index alignment corrected)')
    print('=' * 80)

    try:
        uel_fixed = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_fixed
        )

        uel_fixed.run(
            experiment_name='tradeline_fixed_compare',
            n_permutations=n_perms,
            prep_each_round=True
        )

        results['fixed'] = {
            'log': uel_fixed.experiment_log,
            'status': 'completed'
        }
        print(f'Fixed SFM completed: {len(uel_fixed.experiment_log)} permutations')

    except Exception as e:
        print(f'ERROR running fixed SFM: {e}')
        import traceback
        traceback.print_exc()
        results['fixed'] = {'status': 'error', 'error': str(e)}

    # Compare results
    print('\n' + '=' * 80)
    print('COMPARISON RESULTS')
    print('=' * 80)

    if results.get('original', {}).get('status') == 'completed' and \
       results.get('fixed', {}).get('status') == 'completed':

        log_orig = results['original']['log']
        log_fixed = results['fixed']['log']

        # Key trading metrics
        metrics = ['trading_return_net_pct', 'trading_win_rate_pct', 'trading_trades_count']

        print(f'\n{"Metric":<30} {"Original":<15} {"Fixed":<15} {"Diff":<15}')
        print('-' * 75)

        comparison_data = []

        for metric in metrics:
            if metric in log_orig.columns and metric in log_fixed.columns:
                orig_mean = log_orig[metric].mean()
                fixed_mean = log_fixed[metric].mean()
                diff = fixed_mean - orig_mean

                print(f'{metric:<30} {orig_mean:<15.4f} {fixed_mean:<15.4f} {diff:<+15.4f}')

                comparison_data.append({
                    'metric': metric,
                    'original_mean': orig_mean,
                    'fixed_mean': fixed_mean,
                    'diff': diff,
                    'original_std': log_orig[metric].std(),
                    'fixed_std': log_fixed[metric].std()
                })

        # Best permutation comparison
        print('\n' + '-' * 75)
        print('BEST PERMUTATION BY TRADING RETURN:')
        print('-' * 75)

        if 'trading_return_net_pct' in log_orig.columns:
            best_orig_idx = log_orig['trading_return_net_pct'].arg_max()
            best_orig = log_orig.row(best_orig_idx, named=True)
            print(f"\nOriginal Best:")
            print(f"  Return: {best_orig.get('trading_return_net_pct', 0):.2f}%")
            print(f"  Win Rate: {best_orig.get('trading_win_rate_pct', 0):.2f}%")
            print(f"  Trades: {best_orig.get('trading_trades_count', 0):.0f}")
            print(f"  lookahead_hours: {best_orig.get('lookahead_hours', 'N/A')}")
            print(f"  max_duration_hours: {best_orig.get('max_duration_hours', 'N/A')}")
            print(f"  max_hold_hours: {best_orig.get('max_hold_hours', 'N/A')}")

        if 'trading_return_net_pct' in log_fixed.columns:
            best_fixed_idx = log_fixed['trading_return_net_pct'].arg_max()
            best_fixed = log_fixed.row(best_fixed_idx, named=True)
            print(f"\nFixed Best:")
            print(f"  Return: {best_fixed.get('trading_return_net_pct', 0):.2f}%")
            print(f"  Win Rate: {best_fixed.get('trading_win_rate_pct', 0):.2f}%")
            print(f"  Trades: {best_fixed.get('trading_trades_count', 0):.0f}")
            print(f"  lookahead_hours: {best_fixed.get('lookahead_hours', 'N/A')}")
            print(f"  max_duration_hours: {best_fixed.get('max_duration_hours', 'N/A')}")
            print(f"  max_hold_hours: {best_fixed.get('max_hold_hours', 'N/A')}")

        # Save comparison results
        comparison_df = pl.DataFrame(comparison_data)
        comparison_df.write_csv('/Users/beyondsyntax/Loop/catalysis/original_vs_fixed_comparison.csv')
        print(f'\nComparison saved to: catalysis/original_vs_fixed_comparison.csv')

        # Save full logs
        log_orig.write_csv('/Users/beyondsyntax/Loop/catalysis/original_sfm_full_log.csv')
        log_fixed.write_csv('/Users/beyondsyntax/Loop/catalysis/fixed_sfm_full_log.csv')
        print(f'Full logs saved to: catalysis/original_sfm_full_log.csv, catalysis/fixed_sfm_full_log.csv')

    else:
        print('Cannot compare - one or both experiments failed')
        for name, res in results.items():
            print(f'  {name}: {res.get("status", "unknown")} - {res.get("error", "")}')

    print('\n' + '=' * 80)
    print('COMPARISON COMPLETE')
    print('=' * 80)

    return results


if __name__ == '__main__':
    run_comparison()
