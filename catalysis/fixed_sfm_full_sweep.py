#!/usr/bin/env python3
'''
Full param sweep for fixed tradeline_long_binary SFM
500 permutations on 3 months of 2024 data
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import catalysis.tradeline_long_binary_fixed as sfm_fixed
import polars as pl

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
START_DATE = '2024-01-01'
N_ROWS = 2160  # 90 days * 24 hours = 3 months
N_PERMUTATIONS = 500

print('=' * 80)
print('TRADELINE LONG BINARY FIXED - FULL PARAM SWEEP')
print('=' * 80)
print(f'Period: {START_DATE}, {N_ROWS} hourly candles (3 months)')
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

uel = loop.UniversalExperimentLoop(
    data=data,
    single_file_model=sfm_fixed
)

uel.run(
    experiment_name='tradeline_fixed_full_sweep',
    n_permutations=N_PERMUTATIONS,
    prep_each_round=True
)

log = uel.experiment_log
print(f'\nCompleted: {len(log)} permutations')

# Results summary
print('\n' + '=' * 80)
print('RESULTS SUMMARY')
print('=' * 80)

metrics = ['trading_return_net_pct', 'trading_win_rate_pct', 'trading_trades_count']

print(f'\n{"Metric":<30} {"Mean":<12} {"Std":<12} {"Min":<12} {"Max":<12}')
print('-' * 78)

for m in metrics:
    if m in log.columns:
        print(f'{m:<30} {log[m].mean():<12.4f} {log[m].std():<12.4f} {log[m].min():<12.4f} {log[m].max():<12.4f}')

# Best permutation
print('\n' + '=' * 80)
print('BEST PERMUTATION BY TRADING RETURN')
print('=' * 80)

if 'trading_return_net_pct' in log.columns:
    best_idx = log['trading_return_net_pct'].arg_max()
    best = log.row(best_idx, named=True)

    print(f'\nReturn: {best.get("trading_return_net_pct", 0):.4f}%')
    print(f'Win Rate: {best.get("trading_win_rate_pct", 0):.2f}%')
    print(f'Trades: {best.get("trading_trades_count", 0):.0f}')

    print('\nParams:')
    param_cols = [
        'quantile_threshold', 'min_height_pct', 'max_duration_hours',
        'lookahead_hours', 'long_threshold_percentile', 'loser_timeout_hours',
        'max_hold_hours', 'confidence_threshold', 'position_size',
        'min_stop_loss', 'max_stop_loss', 'atr_stop_multiplier',
        'trailing_activation', 'trailing_distance', 'default_atr_pct',
        'num_leaves', 'learning_rate', 'feature_fraction',
        'bagging_fraction', 'min_child_samples', 'lambda_l1', 'lambda_l2'
    ]

    for p in param_cols:
        if p in best:
            print(f'  {p}: {best[p]}')

# Save results (exclude object columns)
print('\n' + '=' * 80)
print('SAVING RESULTS')
print('=' * 80)

# Filter to serializable columns only
serializable_cols = []
for col in log.columns:
    dtype = log[col].dtype
    if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Utf8, pl.Boolean]:
        serializable_cols.append(col)

log_clean = log.select(serializable_cols)
log_clean.write_csv('/Users/beyondsyntax/Loop/catalysis/fixed_sfm_full_sweep_results.csv')
print(f'Results saved to: catalysis/fixed_sfm_full_sweep_results.csv')
print(f'Columns saved: {len(serializable_cols)}')

print('\n' + '=' * 80)
print('SWEEP COMPLETE')
print('=' * 80)
