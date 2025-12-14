#!/usr/bin/env python3
'''
Multi-window LightGBM vs TabPFN comparison
6-month windows across 2023-2024
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import catalysis.tradeline_long_binary_fixed as sfm_lightgbm
import catalysis.tradeline_long_binary_tabpfn as sfm_tabpfn
import polars as pl
from datetime import datetime

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
N_ROWS_6MO = 4320  # 180 days * 24 hours = 6 months

# 6-month windows
WINDOWS = [
    {'name': '2023_H1', 'start': '2023-01-01'},
    {'name': '2023_H2', 'start': '2023-07-01'},
    {'name': '2024_H1', 'start': '2024-01-01'},
]

# Shared params - 32 perms for ~1.5 hour total runtime
SHARED_PARAMS = {
    'quantile_threshold': [0.70, 0.75],
    'min_height_pct': [0.002, 0.003],
    'max_duration_hours': [48],
    'lookahead_hours': [24, 48],
    'long_threshold_percentile': [75],
    'loser_timeout_hours': [24],
    'max_hold_hours': [48, 72],
    'confidence_threshold': [0.50],
    'position_size': [0.20],
    'min_stop_loss': [0.010],
    'max_stop_loss': [0.040],
    'atr_stop_multiplier': [1.5],
    'trailing_activation': [0.02],
    'trailing_distance': [0.5],
    'default_atr_pct': [0.015],
}

# LightGBM-specific (fixed)
LIGHTGBM_PARAMS = {
    **SHARED_PARAMS,
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

# TabPFN-specific
TABPFN_PARAMS = {
    **SHARED_PARAMS,
    'n_estimators': [8],
    'softmax_temperature': [0.9],
}


def count_permutations(params):
    n = 1
    for v in params.values():
        n *= len(v)
    return n


def run_window(window_name, start_date, n_rows):
    '''Run comparison for a single window'''
    print(f'\n{"="*80}')
    print(f'WINDOW: {window_name}')
    print(f'Start: {start_date}, Rows: {n_rows}')
    print('='*80)

    # Load data
    print('\nLoading data...')
    historical = loop.HistoricalData()
    historical.get_spot_klines(
        kline_size=KLINE_SIZE,
        start_date_limit=start_date,
        n_rows=n_rows
    )
    data = historical.data

    print(f'Loaded {len(data):,} candles')
    print(f'Date range: {data["datetime"].min()} to {data["datetime"].max()}')

    if len(data) < 500:
        print(f'ERROR: Insufficient data ({len(data)} rows)')
        return None

    results = {'window': window_name, 'start_date': start_date}

    # Monkey-patch params
    sfm_lightgbm.params = lambda: LIGHTGBM_PARAMS
    sfm_tabpfn.params = lambda: TABPFN_PARAMS

    n_perms = count_permutations(SHARED_PARAMS)
    print(f'Permutations per model: {n_perms}')

    # Run LightGBM
    print(f'\n--- LightGBM ---')
    try:
        uel_lgb = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_lightgbm
        )
        uel_lgb.run(
            experiment_name=f'lgbm_{window_name}',
            n_permutations=n_perms,
            prep_each_round=True
        )
        log_lgb = uel_lgb.experiment_log

        results['lgbm_return_mean'] = log_lgb['trading_return_net_pct'].mean()
        results['lgbm_return_max'] = log_lgb['trading_return_net_pct'].max()
        results['lgbm_return_std'] = log_lgb['trading_return_net_pct'].std()
        results['lgbm_winrate_mean'] = log_lgb['trading_win_rate_pct'].mean()
        results['lgbm_trades_mean'] = log_lgb['trading_trades_count'].mean()
        results['lgbm_status'] = 'completed'
        results['lgbm_perms'] = len(log_lgb)

        print(f'LightGBM: {len(log_lgb)} perms, best return: {results["lgbm_return_max"]:.2f}%')

    except Exception as e:
        print(f'LightGBM ERROR: {e}')
        results['lgbm_status'] = f'error: {e}'
        results['lgbm_return_mean'] = None
        results['lgbm_return_max'] = None

    # Run TabPFN
    print(f'\n--- TabPFN ---')
    try:
        uel_tabpfn = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_tabpfn
        )
        uel_tabpfn.run(
            experiment_name=f'tabpfn_{window_name}',
            n_permutations=n_perms,
            prep_each_round=True
        )
        log_tabpfn = uel_tabpfn.experiment_log

        results['tabpfn_return_mean'] = log_tabpfn['trading_return_net_pct'].mean()
        results['tabpfn_return_max'] = log_tabpfn['trading_return_net_pct'].max()
        results['tabpfn_return_std'] = log_tabpfn['trading_return_net_pct'].std()
        results['tabpfn_winrate_mean'] = log_tabpfn['trading_win_rate_pct'].mean()
        results['tabpfn_trades_mean'] = log_tabpfn['trading_trades_count'].mean()
        results['tabpfn_status'] = 'completed'
        results['tabpfn_perms'] = len(log_tabpfn)

        print(f'TabPFN: {len(log_tabpfn)} perms, best return: {results["tabpfn_return_max"]:.2f}%')

    except Exception as e:
        print(f'TabPFN ERROR: {e}')
        results['tabpfn_status'] = f'error: {e}'
        results['tabpfn_return_mean'] = None
        results['tabpfn_return_max'] = None

    # Calculate diff
    if results.get('lgbm_return_max') and results.get('tabpfn_return_max'):
        results['diff_return_max'] = results['tabpfn_return_max'] - results['lgbm_return_max']
        results['diff_return_mean'] = results['tabpfn_return_mean'] - results['lgbm_return_mean']
        results['winner'] = 'TabPFN' if results['diff_return_max'] > 0 else 'LightGBM'
    else:
        results['winner'] = 'N/A'

    return results


def run_all_windows():
    print('='*80)
    print('MULTI-WINDOW LIGHTGBM vs TABPFN COMPARISON')
    print('='*80)
    print(f'Windows: {len(WINDOWS)}')
    print(f'Each window: {N_ROWS_6MO} hourly candles (~6 months)')
    print(f'Permutations per model: {count_permutations(SHARED_PARAMS)}')
    print('='*80)

    all_results = []

    for window in WINDOWS:
        result = run_window(
            window_name=window['name'],
            start_date=window['start'],
            n_rows=N_ROWS_6MO
        )
        if result:
            all_results.append(result)

    # Summary
    print('\n' + '='*80)
    print('MULTI-WINDOW SUMMARY')
    print('='*80)

    print(f'\n{"Window":<12} {"LGB Max":<10} {"TabPFN Max":<12} {"Diff":<10} {"Winner":<10}')
    print('-'*60)

    lgb_wins = 0
    tabpfn_wins = 0

    for r in all_results:
        lgb_max = r.get('lgbm_return_max', 0) or 0
        tabpfn_max = r.get('tabpfn_return_max', 0) or 0
        diff = r.get('diff_return_max', 0) or 0
        winner = r.get('winner', 'N/A')

        print(f'{r["window"]:<12} {lgb_max:<10.2f} {tabpfn_max:<12.2f} {diff:<+10.2f} {winner:<10}')

        if winner == 'LightGBM':
            lgb_wins += 1
        elif winner == 'TabPFN':
            tabpfn_wins += 1

    print('-'*60)
    print(f'\nLightGBM wins: {lgb_wins}/{len(all_results)}')
    print(f'TabPFN wins: {tabpfn_wins}/{len(all_results)}')

    overall_winner = 'TabPFN' if tabpfn_wins > lgb_wins else 'LightGBM' if lgb_wins > tabpfn_wins else 'TIE'
    print(f'\nOVERALL WINNER: {overall_winner}')

    # Save results
    results_df = pl.DataFrame(all_results)
    results_df.write_csv('/Users/beyondsyntax/Loop/catalysis/multi_window_comparison.csv')
    print(f'\nResults saved to: catalysis/multi_window_comparison.csv')

    print('\n' + '='*80)
    print('COMPARISON COMPLETE')
    print('='*80)

    return all_results


if __name__ == '__main__':
    run_all_windows()
