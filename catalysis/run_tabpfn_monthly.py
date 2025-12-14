#!/usr/bin/env python3
'''
TabPFN SFM Monthly Runner
Runs TabPFN binary classifier from Feb 2023 to May 2025
One month at a time with logging
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import catalysis.tradeline_long_binary_tabpfn as sfm_tabpfn
import polars as pl
from datetime import datetime
import sys

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
N_ROWS_MONTH = 720  # 30 days * 24 hours
N_PERMUTATIONS = 20

# Generate monthly windows from Feb 2023 to May 2025
MONTHS = []
for year in [2023, 2024, 2025]:
    if year == 2023:
        start_month = 2  # Feb
        end_month = 12
    elif year == 2024:
        start_month = 1
        end_month = 12
    elif year == 2025:
        start_month = 1
        end_month = 5  # May

    for month in range(start_month, end_month + 1):
        MONTHS.append({
            'name': f'{year}_{month:02d}',
            'start': f'{year}-{month:02d}-01'
        })


def run_month(month_name, start_date, n_rows, n_perms):
    '''Run TabPFN for a single month'''
    print(f'\n{"="*80}')
    print(f'MONTH: {month_name}')
    print(f'Start: {start_date}, Rows: {n_rows}, Perms: {n_perms}')
    print('='*80)

    results = {
        'month': month_name,
        'start_date': start_date,
        'n_rows_requested': n_rows,
        'n_permutations': n_perms
    }

    # Load data
    print('\nLoading data...')
    try:
        historical = loop.HistoricalData()
        historical.get_spot_klines(
            kline_size=KLINE_SIZE,
            start_date_limit=start_date,
            n_rows=n_rows
        )
        data = historical.data

        results['n_rows_loaded'] = len(data)
        results['date_min'] = str(data['datetime'].min())
        results['date_max'] = str(data['datetime'].max())

        print(f'Loaded {len(data):,} candles')
        print(f'Date range: {data["datetime"].min()} to {data["datetime"].max()}')

        if len(data) < 500:
            print(f'WARNING: Low data ({len(data)} rows), may affect results')

    except Exception as e:
        print(f'DATA LOAD ERROR: {e}')
        results['status'] = f'data_error: {e}'
        return results

    # Run TabPFN
    print(f'\nRunning TabPFN with {n_perms} permutations...')
    try:
        uel = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_tabpfn
        )
        uel.run(
            experiment_name=f'tabpfn_{month_name}',
            n_permutations=n_perms,
            prep_each_round=True
        )
        log = uel.experiment_log

        results['perms_completed'] = len(log)
        results['return_mean'] = float(log['trading_return_net_pct'].mean())
        results['return_max'] = float(log['trading_return_net_pct'].max())
        results['return_min'] = float(log['trading_return_net_pct'].min())
        results['return_std'] = float(log['trading_return_net_pct'].std())
        results['winrate_mean'] = float(log['trading_win_rate_pct'].mean())
        results['trades_mean'] = float(log['trading_trades_count'].mean())
        results['status'] = 'completed'

        print(f'\nResults for {month_name}:')
        print(f'  Return: mean={results["return_mean"]:.2f}%, max={results["return_max"]:.2f}%, min={results["return_min"]:.2f}%')
        print(f'  Win rate: {results["winrate_mean"]:.1f}%')
        print(f'  Trades: {results["trades_mean"]:.1f}')

    except Exception as e:
        print(f'UEL ERROR: {e}')
        results['status'] = f'uel_error: {e}'
        import traceback
        traceback.print_exc()

    return results


def main():
    print('='*80)
    print('TABPFN SFM MONTHLY RUNNER')
    print('='*80)
    print(f'Period: Feb 2023 to May 2025 ({len(MONTHS)} months)')
    print(f'Rows per month: {N_ROWS_MONTH}')
    print(f'Permutations per month: {N_PERMUTATIONS}')
    print('='*80)

    all_results = []

    for i, month in enumerate(MONTHS, 1):
        print(f'\n>>> Progress: {i}/{len(MONTHS)} <<<')

        result = run_month(
            month_name=month['name'],
            start_date=month['start'],
            n_rows=N_ROWS_MONTH,
            n_perms=N_PERMUTATIONS
        )
        all_results.append(result)

        # Save intermediate results after each month
        results_df = pl.DataFrame(all_results)
        results_df.write_csv('/Users/beyondsyntax/Loop/catalysis/tabpfn_monthly_results.csv')
        print(f'Intermediate results saved ({i}/{len(MONTHS)} months)')

    # Final summary
    print('\n' + '='*80)
    print('FINAL SUMMARY')
    print('='*80)

    completed = [r for r in all_results if r.get('status') == 'completed']

    if completed:
        print(f'\nCompleted: {len(completed)}/{len(all_results)} months')

        returns = [r['return_mean'] for r in completed]
        max_returns = [r['return_max'] for r in completed]

        print(f'\nOverall Statistics:')
        print(f'  Mean return (across months): {sum(returns)/len(returns):.2f}%')
        print(f'  Best single month max: {max(max_returns):.2f}%')
        print(f'  Worst single month mean: {min(returns):.2f}%')

        # Show best/worst months
        sorted_by_mean = sorted(completed, key=lambda x: x['return_mean'], reverse=True)

        print(f'\nTop 5 months by mean return:')
        for r in sorted_by_mean[:5]:
            print(f'  {r["month"]}: {r["return_mean"]:.2f}%')

        print(f'\nBottom 5 months by mean return:')
        for r in sorted_by_mean[-5:]:
            print(f'  {r["month"]}: {r["return_mean"]:.2f}%')

    print(f'\nFinal results saved to: catalysis/tabpfn_monthly_results.csv')
    print('\n' + '='*80)
    print('RUN COMPLETE')
    print('='*80)

    return all_results


if __name__ == '__main__':
    main()
