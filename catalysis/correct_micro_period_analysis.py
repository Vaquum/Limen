#!/usr/bin/env python3
"""
Correct Micro-Period Sensitivity Analysis
Train model on full period, then test on micro-periods to evaluate temporal stability
"""
import loop
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any
from loop.tests.utils.get_data import get_klines_data

def create_test_windows(data: pl.DataFrame, window_days: int, max_windows: int = None) -> List[pl.DataFrame]:
    """
    Create micro-period test windows from data

    Args:
        data: Full dataset
        window_days: Size of each test window in days
        max_windows: Maximum windows to create (None = no limit)

    Returns:
        List of test windows
    """
    # Sort by datetime
    data_sorted = data.sort('datetime')

    # Handle datetime types
    if data_sorted.schema['datetime'] != pl.Datetime:
        data_sorted = data_sorted.with_columns(
            pl.col('datetime').str.strptime(pl.Datetime)
        )

    start_date = data_sorted['datetime'].min()
    end_date = data_sorted['datetime'].max()

    # Convert to python datetime for calculations
    start_py = start_date.to_py() if hasattr(start_date, 'to_py') else start_date
    end_py = end_date.to_py() if hasattr(end_date, 'to_py') else end_date
    total_duration = (end_py - start_py).total_seconds() / 86400  # days

    if total_duration <= window_days:
        return [data_sorted]

    # Use 1-day stride for comprehensive micro-period coverage
    stride_days = 1

    if max_windows is not None:
        stride_days = max(1, int((total_duration - window_days) / max(max_windows - 1, 1)))

    windows = []
    current_start = start_py
    window_count = 0

    while current_start < end_py and (max_windows is None or window_count < max_windows):
        window_end = current_start + timedelta(days=window_days)

        if window_end > end_py:
            window_end = end_py

        # Filter data for this window
        window_data = data_sorted.filter(
            (pl.col('datetime') >= current_start) &
            (pl.col('datetime') <= window_end)
        )

        # Only add windows with some data
        if len(window_data) >= 10:
            windows.append(window_data)
            window_count += 1

        current_start += timedelta(days=stride_days)

    return windows

def train_model_on_full_data(data: pl.DataFrame, sfm_model) -> Any:
    """
    Train SFM model on full dataset

    Args:
        data: Full dataset for training
        sfm_model: SFM model function

    Returns:
        Trained model object
    """
    print("Training model on full dataset...")

    # Create UEL instance for training
    uel = loop.UniversalExperimentLoop(
        data=data,
        single_file_model=sfm_model
    )

    # Train with single permutation
    uel.run(
        experiment_name='full_dataset_training',
        n_permutations=1,
        prep_each_round=True
    )

    if len(uel.experiment_log) == 0:
        raise ValueError("Failed to train model on full dataset")

    # Get trained model info
    result_row = uel.experiment_log.row(0, named=True)
    print(f"Training complete. AUC: {result_row.get('auc', 'N/A'):.4f}")

    return uel

def test_model_on_window(trained_uel, window_data: pl.DataFrame, verbose: bool = False) -> Dict[str, float]:
    """
    Test trained model on micro-period window

    Args:
        trained_uel: Trained UEL instance
        window_data: Test window data
        verbose: Show progress

    Returns:
        Performance metrics on test window
    """
    try:
        if verbose:
            print(f"    Testing on window with {len(window_data)} rows...")

        # Create new UEL instance for testing with same model
        test_uel = loop.UniversalExperimentLoop(
            data=window_data,
            single_file_model=trained_uel.single_file_model
        )

        # Test with trained model parameters
        # Get parameters from trained model
        if len(trained_uel.experiment_log) > 0:
            trained_params = trained_uel.experiment_log.row(0, named=True)

            # Create parameter function using trained parameters
            def use_trained_params():
                # Extract relevant parameters from trained model
                param_dict = {}
                for key, value in trained_params.items():
                    if key.endswith('_param') or key in ['learning_rate', 'num_leaves', 'n_estimators']:
                        param_dict[key] = [value] if not isinstance(value, list) else value
                return param_dict

            test_uel.run(
                experiment_name=f'micro_test_{len(window_data)}_rows',
                n_permutations=1,
                prep_each_round=True,
                params=use_trained_params
            )
        else:
            # Fallback to default parameters
            test_uel.run(
                experiment_name=f'micro_test_{len(window_data)}_rows',
                n_permutations=1,
                prep_each_round=True
            )

        # Extract test metrics
        if len(test_uel.experiment_log) > 0:
            result = test_uel.experiment_log.row(0, named=True)

            metrics = {
                'auc': result.get('auc', np.nan),
                'precision': result.get('precision', np.nan),
                'recall': result.get('recall', np.nan),
                'f1': result.get('f1', np.nan),
                'trading_return_net_pct': result.get('trading_return_net_pct', np.nan),
                'trading_win_rate_pct': result.get('trading_win_rate_pct', np.nan),
                'win_loss_ratio': abs(result.get('trading_avg_win', np.nan) / result.get('trading_avg_loss', 1)) if result.get('trading_avg_loss') != 0 else np.nan
            }

            if verbose and not np.isnan(metrics['auc']):
                print(f"    Test AUC: {metrics['auc']:.4f}")

            return metrics
        else:
            if verbose:
                print("    No test results obtained")
            return {k: np.nan for k in ['auc', 'precision', 'recall', 'f1', 'trading_return_net_pct', 'trading_win_rate_pct', 'win_loss_ratio']}

    except Exception as e:
        if verbose:
            print(f"    Test error: {str(e)[:100]}...")
        return {k: np.nan for k in ['auc', 'precision', 'recall', 'f1', 'trading_return_net_pct', 'trading_win_rate_pct', 'win_loss_ratio']}

def calculate_stability_metrics(values: List[float]) -> Dict[str, float]:
    """Calculate stability metrics for a series of values"""
    valid_values = np.array([v for v in values if not np.isnan(v)])

    if len(valid_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'coeff_var': np.nan,
            'percentile_5': np.nan,
            'percentile_95': np.nan,
            'range': np.nan,
            'valid_windows': 0
        }

    mean_val = np.mean(valid_values)
    std_val = np.std(valid_values)

    return {
        'mean': mean_val,
        'std': std_val,
        'coeff_var': std_val / abs(mean_val) if mean_val != 0 else np.inf,
        'percentile_5': np.percentile(valid_values, 5),
        'percentile_95': np.percentile(valid_values, 95),
        'range': np.max(valid_values) - np.min(valid_values),
        'valid_windows': len(valid_values)
    }

def correct_micro_period_analysis(data: pl.DataFrame,
                                 sfm_model,
                                 test_window_days: List[int] = [1, 2, 3],
                                 max_test_windows: int = None,
                                 verbose: bool = False) -> Dict[str, Any]:
    """
    Correct micro-period sensitivity analysis

    Args:
        data: Full dataset
        sfm_model: SFM model to analyze
        test_window_days: List of micro-period sizes to test
        max_test_windows: Maximum test windows per size (None = no limit)
        verbose: Show progress

    Returns:
        Analysis results
    """
    print("=== CORRECT MICRO-PERIOD SENSITIVITY ANALYSIS ===")
    print("Approach: Train on full data, test on micro-periods")

    # Step 1: Train model on full dataset
    trained_model = train_model_on_full_data(data, sfm_model)

    # Step 2: Test on micro-periods
    results = {}

    for window_days in test_window_days:
        print(f"\nTesting on {window_days}-day micro-periods...")

        # Create test windows
        test_windows = create_test_windows(data, window_days, max_test_windows)
        print(f"Created {len(test_windows)} test windows")

        if len(test_windows) == 0:
            print(f"No valid test windows for {window_days}-day analysis")
            continue

        # Test on each window
        all_metrics = {
            'auc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'trading_return_net_pct': [],
            'trading_win_rate_pct': [],
            'win_loss_ratio': []
        }

        valid_tests = 0

        for i, window in enumerate(test_windows, 1):
            if verbose:
                print(f"  Testing window {i}/{len(test_windows)} ({len(window)} rows)...")

            window_metrics = test_model_on_window(trained_model, window, verbose)

            # Check for valid results
            has_valid = any(not np.isnan(v) for v in window_metrics.values())
            if has_valid:
                valid_tests += 1

            # Store all results (including NaNs for completeness)
            for metric, value in window_metrics.items():
                all_metrics[metric].append(value)

        print(f"  Completed: {len(test_windows)} windows tested, {valid_tests} with valid results")

        # Calculate stability metrics
        stability_results = {}
        for metric, values in all_metrics.items():
            stability_results[metric] = calculate_stability_metrics(values)

        results[f"{window_days}d"] = {
            'raw_metrics': all_metrics,
            'stability': stability_results,
            'total_windows': len(test_windows),
            'valid_windows': valid_tests
        }

    return results

if __name__ == "__main__":
    print("Loading full dataset...")
    data = get_klines_data()
    print(f"Dataset size: {len(data)} rows")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")

    # Run correct micro-period analysis
    results = correct_micro_period_analysis(
        data=data,
        sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
        test_window_days=[1, 2, 3],  # True micro-periods
        max_test_windows=100,  # Limit for reasonable runtime
        verbose=True
    )

    # Display results
    print("\n" + "="*80)
    print("MICRO-PERIOD SENSITIVITY RESULTS")
    print("="*80)

    for window_key, window_results in results.items():
        print(f"\n{window_key.upper()} TEST WINDOWS:")
        print(f"  Total windows tested: {window_results.get('total_windows', 0)}")
        print(f"  Windows with valid results: {window_results.get('valid_windows', 0)}")

        if window_results.get('valid_windows', 0) > 0:
            print("  Performance stability:")
            key_metrics = ['auc', 'trading_return_net_pct', 'win_loss_ratio']

            for metric in key_metrics:
                if metric in window_results['stability']:
                    stats = window_results['stability'][metric]
                    mean_val = stats.get('mean')
                    if mean_val is not None and not np.isnan(mean_val):
                        print(f"    {metric}: mean={mean_val:.4f}, std={stats.get('std', 0):.4f}, coeff_var={stats.get('coeff_var', 0):.4f}")
                    else:
                        print(f"    {metric}: No valid data")
        else:
            print("  No valid results - micro-periods may be too small")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("Model trained once on full data, then tested on micro-periods")
    print("This approach properly evaluates temporal stability")
    print(f"{'='*80}")