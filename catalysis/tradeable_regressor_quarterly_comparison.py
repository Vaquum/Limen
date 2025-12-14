'''
Quarterly comparison: LightGBM vs TabPFN Tradeable Regressor across 2023-2024.
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import loop.sfm.lightgbm.tradeable_regressor as sfm_lightgbm
import loop.sfm.tabpfn.tradeable_regressor as sfm_tabpfn
import polars as pl
import numpy as np

KLINE_SIZE = 900  # 15 min
N_ROWS_3MO = 8760  # ~3 months of 15-min candles

QUARTERS = [
    {'name': '2023_Q1', 'start': '2023-01-01'},
    {'name': '2023_Q2', 'start': '2023-04-01'},
    {'name': '2023_Q3', 'start': '2023-07-01'},
    {'name': '2023_Q4', 'start': '2023-10-01'},
    {'name': '2024_Q1', 'start': '2024-01-01'},
    {'name': '2024_Q2', 'start': '2024-04-01'},
    {'name': '2024_Q3', 'start': '2024-07-01'},
]


def run_quarter(quarter_name, start_date, n_rows):
    print(f'\n{"="*80}')
    print(f'QUARTER: {quarter_name}')
    print(f'Start: {start_date}, Rows: {n_rows}')
    print('='*80)

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

    if len(data) < 1000:
        print(f'ERROR: Insufficient data ({len(data)} rows)')
        return None

    results = {'quarter': quarter_name, 'start_date': start_date}

    print(f'\n--- LightGBM Tradeable Regressor ---')
    try:
        uel_lgb = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_lightgbm
        )
        uel_lgb.run(
            experiment_name=f'lgbm_tr_{quarter_name}',
            n_permutations=2,
            prep_each_round=True
        )
        log_lgb = uel_lgb.experiment_log

        results['lgbm_rmse_mean'] = log_lgb['val_rmse'].mean()
        results['lgbm_rmse_min'] = log_lgb['val_rmse'].min()
        results['lgbm_status'] = 'completed'
        results['lgbm_perms'] = len(log_lgb)

        print(f'LightGBM: {len(log_lgb)} perms, best RMSE: {results["lgbm_rmse_min"]:.6f}')

    except Exception as e:
        print(f'LightGBM ERROR: {e}')
        results['lgbm_status'] = f'error: {e}'
        results['lgbm_rmse_mean'] = None
        results['lgbm_rmse_min'] = None

    print(f'\n--- TabPFN Tradeable Regressor ---')
    try:
        uel_tabpfn = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_tabpfn
        )
        uel_tabpfn.run(
            experiment_name=f'tabpfn_tr_{quarter_name}',
            n_permutations=2,
            prep_each_round=True
        )
        log_tabpfn = uel_tabpfn.experiment_log

        results['tabpfn_rmse_mean'] = log_tabpfn['val_rmse'].mean()
        results['tabpfn_rmse_min'] = log_tabpfn['val_rmse'].min()
        results['tabpfn_correlation_mean'] = log_tabpfn['test_correlation'].mean() if 'test_correlation' in log_tabpfn.columns else None
        results['tabpfn_status'] = 'completed'
        results['tabpfn_perms'] = len(log_tabpfn)

        print(f'TabPFN: {len(log_tabpfn)} perms, best RMSE: {results["tabpfn_rmse_min"]:.6f}')

    except Exception as e:
        print(f'TabPFN ERROR: {e}')
        results['tabpfn_status'] = f'error: {e}'
        results['tabpfn_rmse_mean'] = None
        results['tabpfn_rmse_min'] = None

    if results.get('lgbm_rmse_min') and results.get('tabpfn_rmse_min'):
        results['rmse_diff'] = results['tabpfn_rmse_min'] - results['lgbm_rmse_min']
        results['winner'] = 'TabPFN' if results['rmse_diff'] < 0 else 'LightGBM'
    else:
        results['winner'] = 'N/A'

    return results


def run_all_quarters():
    print('='*80)
    print('QUARTERLY TRADEABLE REGRESSOR COMPARISON')
    print('LightGBM vs TabPFN')
    print('='*80)
    print(f'Quarters: {len(QUARTERS)}')
    print(f'Each quarter: {N_ROWS_3MO} 15-min candles (~3 months)')
    print('='*80)

    all_results = []

    for quarter in QUARTERS:
        result = run_quarter(
            quarter_name=quarter['name'],
            start_date=quarter['start'],
            n_rows=N_ROWS_3MO
        )
        if result:
            all_results.append(result)

    print('\n' + '='*80)
    print('QUARTERLY SUMMARY')
    print('='*80)

    print(f'\n{"Quarter":<12} {"LGB RMSE":<12} {"TabPFN RMSE":<14} {"Diff":<12} {"Winner":<10}')
    print('-'*60)

    lgb_wins = 0
    tabpfn_wins = 0

    for r in all_results:
        lgb_rmse = r.get('lgbm_rmse_min', 0) or 0
        tabpfn_rmse = r.get('tabpfn_rmse_min', 0) or 0
        diff = r.get('rmse_diff', 0) or 0
        winner = r.get('winner', 'N/A')

        print(f'{r["quarter"]:<12} {lgb_rmse:<12.6f} {tabpfn_rmse:<14.6f} {diff:<+12.6f} {winner:<10}')

        if winner == 'LightGBM':
            lgb_wins += 1
        elif winner == 'TabPFN':
            tabpfn_wins += 1

    print('-'*60)
    print(f'\nLightGBM wins: {lgb_wins}/{len(all_results)}')
    print(f'TabPFN wins: {tabpfn_wins}/{len(all_results)}')

    overall_winner = 'TabPFN' if tabpfn_wins > lgb_wins else 'LightGBM' if lgb_wins > tabpfn_wins else 'TIE'
    print(f'\nOVERALL WINNER: {overall_winner}')

    results_df = pl.DataFrame(all_results)
    results_df.write_csv('/Users/beyondsyntax/Loop/catalysis/tradeable_regressor_quarterly_15min.csv')
    print(f'\nResults saved to: catalysis/tradeable_regressor_quarterly_15min.csv')

    print('\n' + '='*80)
    print('COMPARISON COMPLETE')
    print('='*80)

    return all_results


if __name__ == '__main__':
    run_all_quarters()
