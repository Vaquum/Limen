'''
Test script for tradeline_long_binary_fixed SFM
Tests the fixed version against 6 months of 2024 data, loading 1 month at a time
'''
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import loop
import catalysis.tradeline_long_binary_fixed as sfm_fixed

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
ROWS_PER_MONTH = 750  # ~30 days * 24 hours + buffer
N_PERMUTATIONS = 10  # Number of parameter permutations to test

# Test periods: Jan 2024 - Jun 2024 (6 months)
TEST_MONTHS = [
    ('2024-01', '2024-01-01', '2024-01-31'),
    ('2024-02', '2024-02-01', '2024-02-29'),
    ('2024-03', '2024-03-01', '2024-03-31'),
    ('2024-04', '2024-04-01', '2024-04-30'),
    ('2024-05', '2024-05-01', '2024-05-31'),
    ('2024-06', '2024-06-01', '2024-06-30'),
]

print('=' * 80)
print('TRADELINE LONG BINARY FIXED - 6 MONTH TEST')
print('=' * 80)
print(f'Testing fixed SFM with bug resolution:')
print(f'  - Bug: Index misalignment between full dataset and test predictions')
print(f'  - Fix: Only test split data passed to trading simulation')
print(f'Kline size: {KLINE_SIZE}s (1 hour)')
print(f'Rows per month: ~{ROWS_PER_MONTH}')
print(f'Permutations per month: {N_PERMUTATIONS}')
print('=' * 80)

historical = loop.HistoricalData()
results_summary = []

for month_name, start_date, end_date in TEST_MONTHS:
    print(f'\n{"=" * 80}')
    print(f'MONTH: {month_name}')
    print(f'Period: {start_date} to {end_date}')
    print(f'{"=" * 80}')

    # Load data for this month
    print(f'\nLoading {ROWS_PER_MONTH} hourly candles starting from {start_date}...')
    try:
        historical.get_spot_klines(
            kline_size=KLINE_SIZE,
            start_date_limit=start_date,
            n_rows=ROWS_PER_MONTH
        )
        data = historical.data

        print(f'Loaded {len(data):,} candles')
        print(f'Date range: {data["datetime"].min()} to {data["datetime"].max()}')

        if len(data) < 100:
            print(f'WARNING: Insufficient data ({len(data)} rows), skipping month')
            results_summary.append({
                'month': month_name,
                'status': 'skipped',
                'reason': 'insufficient_data',
                'rows': len(data)
            })
            continue

    except Exception as e:
        print(f'ERROR loading data: {e}')
        results_summary.append({
            'month': month_name,
            'status': 'error',
            'reason': str(e),
            'rows': 0
        })
        continue

    # Run UEL experiment for this month
    print(f'\nRunning UEL experiment with {N_PERMUTATIONS} permutations...')
    try:
        uel = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_fixed
        )

        experiment_name = f'tradeline_fixed_{month_name}'

        results = uel.run(
            experiment_name=experiment_name,
            n_permutations=N_PERMUTATIONS,
            prep_each_round=True
        )

        # Check if UEL returned valid results
        if results is None:
            print(f'WARNING: UEL returned None - all permutations likely failed')
            results_summary.append({
                'month': month_name,
                'status': 'all_permutations_failed',
                'reason': 'All permutations failed, likely due to insufficient signals',
                'rows': len(data)
            })
            continue

        # Extract key metrics
        best_perm = results.get('best_permutation', {})
        best_metrics = best_perm.get('metrics', {})

        trading_return = best_metrics.get('trading_return_net_pct', 0.0)
        trading_win_rate = best_metrics.get('trading_win_rate_pct', 0.0)
        trading_trades = best_metrics.get('trading_trades_count', 0)

        print(f'\n{"-" * 80}')
        print(f'RESULTS FOR {month_name}:')
        print(f'{"-" * 80}')
        print(f'Best Trading Return: {trading_return:.2f}%')
        print(f'Best Win Rate: {trading_win_rate:.2f}%')
        print(f'Total Trades: {trading_trades:.0f}')
        print(f'{"-" * 80}')

        results_summary.append({
            'month': month_name,
            'status': 'completed',
            'rows': len(data),
            'trading_return_pct': trading_return,
            'win_rate_pct': trading_win_rate,
            'trades_count': trading_trades,
            'experiment_name': experiment_name
        })

    except Exception as e:
        print(f'ERROR running experiment: {e}')
        import traceback
        traceback.print_exc()
        results_summary.append({
            'month': month_name,
            'status': 'error',
            'reason': str(e),
            'rows': len(data)
        })

# Print final summary
print(f'\n\n{"=" * 80}')
print('FINAL SUMMARY - ALL MONTHS')
print(f'{"=" * 80}')
print(f'\n{"Month":<12} {"Status":<12} {"Rows":<8} {"Return %":<12} {"Win Rate %":<12} {"Trades":<8}')
print('-' * 80)

for result in results_summary:
    month = result['month']
    status = result['status']
    rows = result.get('rows', 0)

    if status == 'completed':
        ret = result.get('trading_return_pct', 0.0)
        win = result.get('win_rate_pct', 0.0)
        trades = result.get('trades_count', 0)
        print(f'{month:<12} {status:<12} {rows:<8} {ret:<12.2f} {win:<12.2f} {trades:<8.0f}')
    else:
        reason = result.get('reason', 'unknown')
        print(f'{month:<12} {status:<12} {rows:<8} {reason}')

print('=' * 80)

# Calculate aggregate statistics for completed months
completed_results = [r for r in results_summary if r['status'] == 'completed']
if completed_results:
    avg_return = sum(r['trading_return_pct'] for r in completed_results) / len(completed_results)
    avg_win_rate = sum(r['win_rate_pct'] for r in completed_results) / len(completed_results)
    total_trades = sum(r['trades_count'] for r in completed_results)

    print(f'\nAGGREGATE STATISTICS ({len(completed_results)} months):')
    print(f'  Average Return: {avg_return:.2f}%')
    print(f'  Average Win Rate: {avg_win_rate:.2f}%')
    print(f'  Total Trades: {total_trades:.0f}')
    print('=' * 80)
else:
    print('\nNo completed months to aggregate')

print('\nTest complete!')
