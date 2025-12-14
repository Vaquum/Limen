#!/usr/bin/env python3
'''
Directional Conditional SFM Comparison: LightGBM vs TabPFN
3 six-month windows across 2023-2024
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import loop.sfm.lightgbm.tradeline_directional_conditional as sfm_lightgbm
import catalysis.tradeline_directional_conditional_tabpfn as sfm_tabpfn
import polars as pl

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
N_ROWS_6MO = 4320  # 180 days * 24 hours = 6 months
N_PERMUTATIONS = 20

# 6-month windows
WINDOWS = [
    {'name': '2023_H1', 'start': '2023-01-01'},
    {'name': '2023_H2', 'start': '2023-07-01'},
    {'name': '2024_H1', 'start': '2024-01-01'},
]

# Shared params for fair comparison - minimal for speed
SHARED_PARAMS = {
    'threshold_pct': [0.010],
    'lookahead_hours': [48],
    'quantile_threshold': [0.75],
    'min_height_pct': [0.003],
    'max_duration_hours': [48],
    'conditional_threshold': [0.7],
    'movement_threshold': [0.3],
    'use_safer': [False],
}

# LightGBM-specific
LIGHTGBM_PARAMS = {
    **SHARED_PARAMS,
    'num_leaves': [31],
    'learning_rate': [0.05],
    'n_estimators': [200],
}

# TabPFN-specific - 2 permutations
TABPFN_PARAMS = {
    **SHARED_PARAMS,
    'n_estimators': [4, 8],
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

    n_perms_lgb = count_permutations(LIGHTGBM_PARAMS)
    n_perms_tabpfn = count_permutations(TABPFN_PARAMS)
    print(f'LightGBM permutations: {n_perms_lgb}')
    print(f'TabPFN permutations: {n_perms_tabpfn}')

    # Run LightGBM
    print(f'\n--- LightGBM Directional Conditional ---')
    try:
        uel_lgb = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_lightgbm
        )
        uel_lgb.run(
            experiment_name=f'lgbm_dc_{window_name}',
            n_permutations=n_perms_lgb,
            prep_each_round=True
        )
        log_lgb = uel_lgb.experiment_log

        results['lgbm_auc_mean'] = float(log_lgb['auc'].mean())
        results['lgbm_auc_max'] = float(log_lgb['auc'].max())
        results['lgbm_precision_mean'] = float(log_lgb['precision'].mean())
        results['lgbm_recall_mean'] = float(log_lgb['recall'].mean())
        results['lgbm_perms'] = len(log_lgb)
        results['lgbm_status'] = 'completed'

        print(f'LightGBM: {len(log_lgb)} perms, AUC mean={results["lgbm_auc_mean"]:.4f}, max={results["lgbm_auc_max"]:.4f}')

    except Exception as e:
        print(f'LightGBM ERROR: {e}')
        import traceback
        traceback.print_exc()
        results['lgbm_status'] = f'error: {e}'
        results['lgbm_auc_mean'] = None
        results['lgbm_auc_max'] = None

    # Run TabPFN
    print(f'\n--- TabPFN Directional Conditional ---')
    try:
        uel_tabpfn = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_tabpfn
        )
        uel_tabpfn.run(
            experiment_name=f'tabpfn_dc_{window_name}',
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
    if results.get('lgbm_auc_max') and results.get('tabpfn_auc_max'):
        results['diff_auc_max'] = results['tabpfn_auc_max'] - results['lgbm_auc_max']
        results['diff_auc_mean'] = results['tabpfn_auc_mean'] - results['lgbm_auc_mean']
        results['winner'] = 'TabPFN' if results['diff_auc_max'] > 0 else 'LightGBM'
    else:
        results['winner'] = 'N/A'

    return results


def main():
    print('='*80)
    print('DIRECTIONAL CONDITIONAL SFM COMPARISON')
    print('LightGBM vs TabPFN')
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
            results_df.write_csv('/Users/beyondsyntax/Loop/catalysis/directional_conditional_comparison.csv')
            print(f'Intermediate results saved ({i}/{len(WINDOWS)} windows)')

    # Summary
    print('\n' + '='*80)
    print('COMPARISON SUMMARY')
    print('='*80)

    print(f'\n{"Window":<12} {"LGB AUC":<12} {"TabPFN AUC":<14} {"Diff":<10} {"Winner":<10}')
    print('-'*60)

    lgb_wins = 0
    tabpfn_wins = 0

    for r in all_results:
        lgb_auc = r.get('lgbm_auc_max', 0) or 0
        tabpfn_auc = r.get('tabpfn_auc_max', 0) or 0
        diff = r.get('diff_auc_max', 0) or 0
        winner = r.get('winner', 'N/A')

        print(f'{r["window"]:<12} {lgb_auc:<12.4f} {tabpfn_auc:<14.4f} {diff:<+10.4f} {winner:<10}')

        if winner == 'LightGBM':
            lgb_wins += 1
        elif winner == 'TabPFN':
            tabpfn_wins += 1

    print('-'*60)
    print(f'\nLightGBM wins: {lgb_wins}/{len(all_results)}')
    print(f'TabPFN wins: {tabpfn_wins}/{len(all_results)}')

    overall_winner = 'TabPFN' if tabpfn_wins > lgb_wins else 'LightGBM' if lgb_wins > tabpfn_wins else 'TIE'
    print(f'\nOVERALL WINNER (by max AUC): {overall_winner}')

    print(f'\nResults saved to: catalysis/directional_conditional_comparison.csv')
    print('\n' + '='*80)
    print('COMPARISON COMPLETE')
    print('='*80)

    return all_results


if __name__ == '__main__':
    main()
