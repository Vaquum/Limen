#!/usr/bin/env python3
'''
Compare LightGBM vs TabPFN tradeline_long_binary SFM
Same parameters, same data, head-to-head comparison
'''
import warnings
warnings.filterwarnings('ignore')

import loop
import catalysis.tradeline_long_binary_fixed as sfm_lightgbm
import catalysis.tradeline_long_binary_tabpfn as sfm_tabpfn
import polars as pl

# Configuration
KLINE_SIZE = 3600  # 1 hour candles
START_DATE = '2024-01-01'
N_ROWS = 2160  # 90 days * 24 hours = 3 months

# Shared params for fair comparison - REDUCED for reasonable runtime
# ~100 permutations each
SHARED_PARAMS = {
    'quantile_threshold': [0.70, 0.75],
    'min_height_pct': [0.003],
    'max_duration_hours': [48],
    'lookahead_hours': [24, 48],
    'long_threshold_percentile': [75],
    'loser_timeout_hours': [24],
    'max_hold_hours': [48, 72],
    'confidence_threshold': [0.45, 0.50],
    'position_size': [0.20],
    'min_stop_loss': [0.010],
    'max_stop_loss': [0.040],
    'atr_stop_multiplier': [1.5, 2.0],
    'trailing_activation': [0.02],
    'trailing_distance': [0.5],
    'default_atr_pct': [0.015],
}

# LightGBM-specific params (fixed to reduce permutations)
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

# TabPFN-specific params
TABPFN_PARAMS = {
    **SHARED_PARAMS,
    'n_estimators': [8],  # TabPFN ensemble size
    'softmax_temperature': [0.9],
}


def create_lightgbm_params():
    return LIGHTGBM_PARAMS


def create_tabpfn_params():
    return TABPFN_PARAMS


def run_comparison():
    print('=' * 80)
    print('LIGHTGBM vs TABPFN COMPARISON')
    print('=' * 80)
    print(f'Period: {START_DATE}, {N_ROWS} hourly candles (3 months)')
    print(f'Kline size: {KLINE_SIZE}s (1 hour)')
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

    # Monkey-patch params functions
    sfm_lightgbm.params = create_lightgbm_params
    sfm_tabpfn.params = create_tabpfn_params

    # Calculate expected permutations
    n_perms = 1
    for v in SHARED_PARAMS.values():
        n_perms *= len(v)
    print(f'\nExpected permutations per model: {n_perms}')

    results = {}

    # Run LightGBM
    print('\n' + '=' * 80)
    print('RUNNING LIGHTGBM SFM')
    print('=' * 80)

    try:
        uel_lgb = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_lightgbm
        )

        uel_lgb.run(
            experiment_name='lightgbm_compare',
            n_permutations=n_perms,
            prep_each_round=True
        )

        results['lightgbm'] = {
            'log': uel_lgb.experiment_log,
            'status': 'completed'
        }
        print(f'LightGBM completed: {len(uel_lgb.experiment_log)} permutations')

    except Exception as e:
        print(f'ERROR running LightGBM: {e}')
        import traceback
        traceback.print_exc()
        results['lightgbm'] = {'status': 'error', 'error': str(e)}

    # Run TabPFN
    print('\n' + '=' * 80)
    print('RUNNING TABPFN SFM')
    print('=' * 80)

    try:
        uel_tabpfn = loop.UniversalExperimentLoop(
            data=data,
            single_file_model=sfm_tabpfn
        )

        uel_tabpfn.run(
            experiment_name='tabpfn_compare',
            n_permutations=n_perms,
            prep_each_round=True
        )

        results['tabpfn'] = {
            'log': uel_tabpfn.experiment_log,
            'status': 'completed'
        }
        print(f'TabPFN completed: {len(uel_tabpfn.experiment_log)} permutations')

    except Exception as e:
        print(f'ERROR running TabPFN: {e}')
        import traceback
        traceback.print_exc()
        results['tabpfn'] = {'status': 'error', 'error': str(e)}

    # Compare results
    print('\n' + '=' * 80)
    print('COMPARISON RESULTS')
    print('=' * 80)

    if results.get('lightgbm', {}).get('status') == 'completed' and \
       results.get('tabpfn', {}).get('status') == 'completed':

        log_lgb = results['lightgbm']['log']
        log_tabpfn = results['tabpfn']['log']

        metrics = ['trading_return_net_pct', 'trading_win_rate_pct', 'trading_trades_count']

        print(f'\n{"Metric":<30} {"LightGBM":<15} {"TabPFN":<15} {"Diff":<15}')
        print('-' * 75)

        comparison_data = []

        for metric in metrics:
            if metric in log_lgb.columns and metric in log_tabpfn.columns:
                lgb_mean = log_lgb[metric].mean()
                tabpfn_mean = log_tabpfn[metric].mean()
                diff = tabpfn_mean - lgb_mean

                print(f'{metric:<30} {lgb_mean:<15.4f} {tabpfn_mean:<15.4f} {diff:<+15.4f}')

                comparison_data.append({
                    'metric': metric,
                    'lightgbm_mean': lgb_mean,
                    'tabpfn_mean': tabpfn_mean,
                    'diff': diff,
                    'lightgbm_std': log_lgb[metric].std(),
                    'tabpfn_std': log_tabpfn[metric].std(),
                    'lightgbm_max': log_lgb[metric].max(),
                    'tabpfn_max': log_tabpfn[metric].max(),
                })

        # Best permutation comparison
        print('\n' + '-' * 75)
        print('BEST PERMUTATION BY TRADING RETURN:')
        print('-' * 75)

        if 'trading_return_net_pct' in log_lgb.columns:
            best_lgb_idx = log_lgb['trading_return_net_pct'].arg_max()
            best_lgb = log_lgb.row(best_lgb_idx, named=True)
            print(f"\nLightGBM Best:")
            print(f"  Return: {best_lgb.get('trading_return_net_pct', 0):.2f}%")
            print(f"  Win Rate: {best_lgb.get('trading_win_rate_pct', 0):.2f}%")
            print(f"  Trades: {best_lgb.get('trading_trades_count', 0):.0f}")

        if 'trading_return_net_pct' in log_tabpfn.columns:
            best_tabpfn_idx = log_tabpfn['trading_return_net_pct'].arg_max()
            best_tabpfn = log_tabpfn.row(best_tabpfn_idx, named=True)
            print(f"\nTabPFN Best:")
            print(f"  Return: {best_tabpfn.get('trading_return_net_pct', 0):.2f}%")
            print(f"  Win Rate: {best_tabpfn.get('trading_win_rate_pct', 0):.2f}%")
            print(f"  Trades: {best_tabpfn.get('trading_trades_count', 0):.0f}")

        # Winner determination
        print('\n' + '=' * 75)
        lgb_best_return = log_lgb['trading_return_net_pct'].max()
        tabpfn_best_return = log_tabpfn['trading_return_net_pct'].max()

        if tabpfn_best_return > lgb_best_return:
            winner = 'TabPFN'
            margin = tabpfn_best_return - lgb_best_return
        else:
            winner = 'LightGBM'
            margin = lgb_best_return - tabpfn_best_return

        print(f'WINNER (Best Return): {winner} by {margin:.2f}%')
        print('=' * 75)

        # Save comparison results
        comparison_df = pl.DataFrame(comparison_data)
        comparison_df.write_csv('/Users/beyondsyntax/Loop/catalysis/lightgbm_vs_tabpfn_comparison.csv')
        print(f'\nComparison saved to: catalysis/lightgbm_vs_tabpfn_comparison.csv')

        # Save full logs (filter to serializable columns only)
        serializable_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Utf8, pl.Boolean]

        lgb_cols = [c for c in log_lgb.columns if log_lgb[c].dtype in serializable_types]
        tabpfn_cols = [c for c in log_tabpfn.columns if log_tabpfn[c].dtype in serializable_types]

        log_lgb.select(lgb_cols).write_csv('/Users/beyondsyntax/Loop/catalysis/lightgbm_compare_log.csv')
        log_tabpfn.select(tabpfn_cols).write_csv('/Users/beyondsyntax/Loop/catalysis/tabpfn_compare_log.csv')
        print(f'Full logs saved to: catalysis/lightgbm_compare_log.csv, catalysis/tabpfn_compare_log.csv')

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
