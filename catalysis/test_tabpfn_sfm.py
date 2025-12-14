#!/usr/bin/env python3
'''
Quick test of the TabPFN SFM
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import catalysis.tradeline_long_binary_tabpfn as sfm_tabpfn

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
START_DATE = '2024-01-01'
N_ROWS = 750  # ~1 month
N_PERMUTATIONS = 5  # Quick test

print('=' * 80)
print('TRADELINE LONG BINARY TABPFN - QUICK TEST')
print('=' * 80)
print(f'Period: {START_DATE}, {N_ROWS} hourly candles')
print(f'Permutations: {N_PERMUTATIONS}')
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

# Run UEL
print(f'\nRunning UEL with {N_PERMUTATIONS} permutations...')
print('(First run will download TabPFN model weights ~400MB)')

uel = loop.UniversalExperimentLoop(
    data=data,
    single_file_model=sfm_tabpfn
)

uel.run(
    experiment_name='tabpfn_quick_test',
    n_permutations=N_PERMUTATIONS,
    prep_each_round=True
)

log = uel.experiment_log
print(f'\nCompleted: {len(log)} permutations')

# Results
print('\n' + '=' * 80)
print('RESULTS')
print('=' * 80)

metrics = ['trading_return_net_pct', 'trading_win_rate_pct', 'trading_trades_count']

print(f'\n{"Metric":<30} {"Mean":<12} {"Std":<12} {"Min":<12} {"Max":<12}')
print('-' * 78)

for m in metrics:
    if m in log.columns:
        print(f'{m:<30} {log[m].mean():<12.4f} {log[m].std():<12.4f} {log[m].min():<12.4f} {log[m].max():<12.4f}')

# Best permutation
if 'trading_return_net_pct' in log.columns:
    best_idx = log['trading_return_net_pct'].arg_max()
    best = log.row(best_idx, named=True)

    print(f'\nBest permutation:')
    print(f'  Return: {best.get("trading_return_net_pct", 0):.4f}%')
    print(f'  Win Rate: {best.get("trading_win_rate_pct", 0):.2f}%')
    print(f'  Trades: {best.get("trading_trades_count", 0):.0f}')
    print(f'  n_estimators: {best.get("n_estimators", "N/A")}')
    print(f'  softmax_temperature: {best.get("softmax_temperature", "N/A")}')

print('\n' + '=' * 80)
print('TEST COMPLETE')
print('=' * 80)
