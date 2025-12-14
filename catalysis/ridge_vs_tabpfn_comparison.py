#!/usr/bin/env python3
'''
Ridge vs TabPFN Regime Classifier Comparison
3 six-month windows across 2023-2024
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import catalysis.regime_classifier_ridge as sfm_ridge
import catalysis.regime_classifier_tabpfn as sfm_tabpfn
import polars as pl

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
N_ROWS_6MO = 4320  # 180 days * 24 hours = 6 months

# 6-month windows
WINDOWS = [
    {'name': '2023_H1', 'start': '2023-01-01'},
    {'name': '2023_H2', 'start': '2023-07-01'},
    {'name': '2024_H1', 'start': '2024-01-01'},
]

# Shared feature params - minimal for speed
SHARED_PARAMS = {
    'roc_period': [4],
    'ppo_fast': [12],
    'ppo_slow': [26],
    'ppo_signal': [9],
    'rsi_period': [14],
    'volatility_window': [24],
    'q': [0.35],
}

# Ridge params (2 perms)
RIDGE_PARAMS = {
    **SHARED_PARAMS,
    'alpha': [2.0, 5.0],
    'pred_threshold': [0.55],
    'use_calibration': [True],
    'calibration_method': ['sigmoid'],
}

# TabPFN params (2 perms)
TABPFN_PARAMS = {
    **SHARED_PARAMS,
    'n_estimators': [4, 8],
    'softmax_temperature': [0.9],
    'pred_threshold': [0.55],
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
    sfm_ridge.params = lambda: RIDGE_PARAMS
    sfm_tabpfn.params = lambda: TABPFN_PARAMS

    n_perms_ridge = count_permutations(RIDGE_PARAMS)
    n_perms_tabpfn = count_permutations(TABPFN_PARAMS)
    print(f'Ridge permutations: {n_perms_ridge}')
    print(f'TabPFN permutations: {n_perms_tabpfn}')

    # Run Ridge
    print(f'\n--- Ridge Classifier ---')
    try:
        uel_ridge = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_ridge
        )
        uel_ridge.run(
            experiment_name=f'ridge_{window_name}',
            n_permutations=n_perms_ridge,
            prep_each_round=True
        )
        log_ridge = uel_ridge.experiment_log

        results['ridge_auc_mean'] = float(log_ridge['auc'].mean())
        results['ridge_auc_max'] = float(log_ridge['auc'].max())
        results['ridge_precision_mean'] = float(log_ridge['precision'].mean())
        results['ridge_recall_mean'] = float(log_ridge['recall'].mean())
        results['ridge_perms'] = len(log_ridge)
        results['ridge_status'] = 'completed'

        print(f'Ridge: {len(log_ridge)} perms, AUC mean={results["ridge_auc_mean"]:.4f}, max={results["ridge_auc_max"]:.4f}')

    except Exception as e:
        print(f'Ridge ERROR: {e}')
        import traceback
        traceback.print_exc()
        results['ridge_status'] = f'error: {e}'
        results['ridge_auc_mean'] = None
        results['ridge_auc_max'] = None

    # Run TabPFN
    print(f'\n--- TabPFN Classifier ---')
    try:
        uel_tabpfn = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_tabpfn
        )
        uel_tabpfn.run(
            experiment_name=f'tabpfn_regime_{window_name}',
            n_permutations=n_perms_tabpfn,
            prep_each_round=True
        )
        log_tabpfn = uel_tabpfn.experiment_log

        results['tabpfn_auc_mean'] = float(log_tabpfn['auc'].mean())
        results['tabpfn_auc_max'] = float(log_tabpfn['auc'].max())
        results['tabpfn_precision_mean'] = float(log_tabpfn['precision'].mean())
        results['tabpfn_recall_mean'] = float(log_tabpfn['recall'].mean())
        results['tabpfn_perms'] = len(log_tabpfn)
        results['tabpfn_status'] = 'completed'

        print(f'TabPFN: {len(log_tabpfn)} perms, AUC mean={results["tabpfn_auc_mean"]:.4f}, max={results["tabpfn_auc_max"]:.4f}')

    except Exception as e:
        print(f'TabPFN ERROR: {e}')
        import traceback
        traceback.print_exc()
        results['tabpfn_status'] = f'error: {e}'
        results['tabpfn_auc_mean'] = None
        results['tabpfn_auc_max'] = None

    # Calculate diff
    if results.get('ridge_auc_max') and results.get('tabpfn_auc_max'):
        results['diff_auc_max'] = results['tabpfn_auc_max'] - results['ridge_auc_max']
        results['diff_auc_mean'] = results['tabpfn_auc_mean'] - results['ridge_auc_mean']
        results['winner'] = 'TabPFN' if results['diff_auc_max'] > 0 else 'Ridge'
    else:
        results['winner'] = 'N/A'

    return results


def main():
    print('='*80)
    print('RIDGE vs TABPFN REGIME CLASSIFIER COMPARISON')
    print('='*80)
    print(f'Windows: {len(WINDOWS)}')
    print(f'Each window: {N_ROWS_6MO} hourly candles (~6 months)')
    print('='*80)

    all_results = []

    for i, window in enumerate(WINDOWS, 1):
        print(f'\n>>> Progress: {i}/{len(WINDOWS)} <<<')
        result = run_window(
            window_name=window['name'],
            start_date=window['start'],
            n_rows=N_ROWS_6MO
        )
        if result:
            all_results.append(result)

            # Save intermediate results
            results_df = pl.DataFrame(all_results)
            results_df.write_csv('/Users/beyondsyntax/Loop/catalysis/ridge_vs_tabpfn_comparison.csv')
            print(f'Intermediate results saved ({i}/{len(WINDOWS)} windows)')

    # Summary
    print('\n' + '='*80)
    print('COMPARISON SUMMARY')
    print('='*80)

    print(f'\n{"Window":<12} {"Ridge AUC":<12} {"TabPFN AUC":<14} {"Diff":<10} {"Winner":<10}')
    print('-'*60)

    ridge_wins = 0
    tabpfn_wins = 0

    for r in all_results:
        ridge_auc = r.get('ridge_auc_max', 0) or 0
        tabpfn_auc = r.get('tabpfn_auc_max', 0) or 0
        diff = r.get('diff_auc_max', 0) or 0
        winner = r.get('winner', 'N/A')

        print(f'{r["window"]:<12} {ridge_auc:<12.4f} {tabpfn_auc:<14.4f} {diff:<+10.4f} {winner:<10}')

        if winner == 'Ridge':
            ridge_wins += 1
        elif winner == 'TabPFN':
            tabpfn_wins += 1

    print('-'*60)
    print(f'\nRidge wins: {ridge_wins}/{len(all_results)}')
    print(f'TabPFN wins: {tabpfn_wins}/{len(all_results)}')

    overall_winner = 'TabPFN' if tabpfn_wins > ridge_wins else 'Ridge' if ridge_wins > tabpfn_wins else 'TIE'
    print(f'\nOVERALL WINNER (by max AUC): {overall_winner}')

    print(f'\nResults saved to: catalysis/ridge_vs_tabpfn_comparison.csv')
    print('\n' + '='*80)
    print('COMPARISON COMPLETE')
    print('='*80)

    return all_results


if __name__ == '__main__':
    main()
