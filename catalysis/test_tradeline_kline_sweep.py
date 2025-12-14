'''
Test tradeline_long_binary_fixed across multiple kline sizes
Fixed lookahead at 24 hours, tests 6 months from 2024, loads 3 months data at a time
'''
from datetime import datetime, timedelta
import loop
import catalysis.tradeline_long_binary_fixed as sfm_fixed

# Kline sizes to test (in seconds)
KLINE_SIZES = {
    '5min': 300,
    '15min': 900,
    '30min': 1800,
    '1h': 3600
}

# Calculate rows needed for ~3 months of data for each kline size
def calculate_rows(kline_size_seconds, months=3):
    days = months * 30
    candles_per_day = (24 * 3600) / kline_size_seconds
    return int(days * candles_per_day)

# Test periods: 6 months in 2024, loading 3 months at a time
TEST_PERIODS = [
    ('Q1-Q2_2024', '2024-01-01', '2024-03-31'),  # 3 months
    ('Q2-Q3_2024', '2024-04-01', '2024-06-30'),  # 3 months
]

N_PERMUTATIONS = 20  # More permutations since we're testing fewer periods

print('=' * 80)
print('TRADELINE LONG BINARY FIXED - KLINE SIZE SWEEP')
print('=' * 80)
print(f'Testing with 24h lookahead across multiple kline sizes')
print(f'Kline sizes: {list(KLINE_SIZES.keys())}')
print(f'Test periods: {len(TEST_PERIODS)} x 3-month windows')
print(f'Permutations per test: {N_PERMUTATIONS}')
print('=' * 80)

historical = loop.HistoricalData()
all_results = []

for kline_name, kline_size in KLINE_SIZES.items():
    rows_needed = calculate_rows(kline_size, months=3)

    print(f'\n{"=" * 80}')
    print(f'KLINE SIZE: {kline_name} ({kline_size}s)')
    print(f'Rows per period: {rows_needed:,}')
    print(f'{"=" * 80}')

    for period_name, start_date, end_date in TEST_PERIODS:
        print(f'\n{"-" * 80}')
        print(f'Period: {period_name} ({start_date} to {end_date})')
        print(f'{"-" * 80}')

        # Load data
        print(f'\nLoading {rows_needed:,} candles of {kline_name} data from {start_date}...')
        try:
            historical.get_spot_klines(
                kline_size=kline_size,
                start_date_limit=start_date,
                n_rows=rows_needed
            )
            data = historical.data

            print(f'Loaded {len(data):,} candles')
            print(f'Date range: {data["datetime"].min()} to {data["datetime"].max()}')

            if len(data) < 500:
                print(f'WARNING: Insufficient data ({len(data)} rows), skipping')
                all_results.append({
                    'kline_size': kline_name,
                    'period': period_name,
                    'status': 'insufficient_data',
                    'rows': len(data)
                })
                continue

        except Exception as e:
            print(f'ERROR loading data: {e}')
            all_results.append({
                'kline_size': kline_name,
                'period': period_name,
                'status': 'load_error',
                'error': str(e),
                'rows': 0
            })
            continue

        # Run UEL experiment
        print(f'\nRunning UEL with {N_PERMUTATIONS} permutations...')
        try:
            uel = loop.UniversalExperimentLoop(
                data=data,
                single_file_model=sfm_fixed
            )

            experiment_name = f'tradeline_fixed_{kline_name}_{period_name}'

            results = uel.run(
                experiment_name=experiment_name,
                n_permutations=N_PERMUTATIONS,
                prep_each_round=True
            )

            if results is None:
                print(f'WARNING: All permutations failed')
                all_results.append({
                    'kline_size': kline_name,
                    'period': period_name,
                    'status': 'all_failed',
                    'rows': len(data)
                })
                continue

            # Extract metrics
            best_perm = results.get('best_permutation', {})
            best_metrics = best_perm.get('metrics', {})
            best_params = best_perm.get('params', {})

            trading_return = best_metrics.get('trading_return_net_pct', 0.0)
            trading_win_rate = best_metrics.get('trading_win_rate_pct', 0.0)
            trading_trades = best_metrics.get('trading_trades_count', 0)

            print(f'\n{"*" * 80}')
            print(f'RESULTS: {kline_name} - {period_name}')
            print(f'{"*" * 80}')
            print(f'Trading Return:  {trading_return:.2f}%')
            print(f'Win Rate:        {trading_win_rate:.2f}%')
            print(f'Total Trades:    {trading_trades:.0f}')
            print(f'Best Params:')
            print(f'  lookahead_hours:       {best_params.get("lookahead_hours", "N/A")}')
            print(f'  max_duration_hours:    {best_params.get("max_duration_hours", "N/A")}')
            print(f'  min_height_pct:        {best_params.get("min_height_pct", "N/A")}')
            print(f'  quantile_threshold:    {best_params.get("quantile_threshold", "N/A")}')
            print(f'{"*" * 80}')

            all_results.append({
                'kline_size': kline_name,
                'period': period_name,
                'status': 'completed',
                'rows': len(data),
                'trading_return_pct': trading_return,
                'win_rate_pct': trading_win_rate,
                'trades_count': trading_trades,
                'lookahead_hours': best_params.get('lookahead_hours'),
                'max_duration_hours': best_params.get('max_duration_hours'),
                'min_height_pct': best_params.get('min_height_pct'),
                'experiment_name': experiment_name
            })

        except Exception as e:
            print(f'ERROR running experiment: {e}')
            import traceback
            traceback.print_exc()
            all_results.append({
                'kline_size': kline_name,
                'period': period_name,
                'status': 'experiment_error',
                'error': str(e),
                'rows': len(data)
            })

# Final summary
print(f'\n\n{"=" * 80}')
print('FINAL SUMMARY - ALL KLINE SIZES')
print(f'{"=" * 80}')
print(f'\n{"Kline":<10} {"Period":<15} {"Status":<15} {"Rows":<10} {"Return %":<12} {"Win %":<10} {"Trades":<8}')
print('-' * 80)

for result in all_results:
    kline = result['kline_size']
    period = result['period']
    status = result['status']
    rows = result.get('rows', 0)

    if status == 'completed':
        ret = result.get('trading_return_pct', 0.0)
        win = result.get('win_rate_pct', 0.0)
        trades = result.get('trades_count', 0)
        print(f'{kline:<10} {period:<15} {status:<15} {rows:<10} {ret:<12.2f} {win:<10.2f} {trades:<8.0f}')
    else:
        reason = result.get('error', status)[:30]
        print(f'{kline:<10} {period:<15} {status:<15} {rows:<10} {reason}')

# Best by kline size
print(f'\n{"=" * 80}')
print('BEST RESULT BY KLINE SIZE')
print(f'{"=" * 80}')

completed = [r for r in all_results if r['status'] == 'completed']
if completed:
    for kline_name in KLINE_SIZES.keys():
        kline_results = [r for r in completed if r['kline_size'] == kline_name]
        if kline_results:
            best = max(kline_results, key=lambda x: x['trading_return_pct'])
            print(f'\n{kline_name}:')
            print(f'  Return:       {best["trading_return_pct"]:.2f}%')
            print(f'  Win Rate:     {best["win_rate_pct"]:.2f}%')
            print(f'  Trades:       {best["trades_count"]:.0f}')
            print(f'  Period:       {best["period"]}')
            print(f'  Lookahead:    {best.get("lookahead_hours", "N/A")}h')
            print(f'  Max Duration: {best.get("max_duration_hours", "N/A")}h')
else:
    print('\nNo completed results')

print(f'\n{"=" * 80}')
print('Test complete!')
print(f'{"=" * 80}')
