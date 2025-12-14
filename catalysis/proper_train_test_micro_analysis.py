#!/usr/bin/env python3
"""
Proper Train-Test Micro-Period Analysis
Train on 2023 data, test on 2024 micro-periods for true out-of-sample evaluation
"""
import loop
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any
from loop.tests.utils.get_data import get_klines_data

def split_data_by_year(data: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """
    Split data by year for proper train-test split

    Args:
        data: Full dataset

    Returns:
        Dictionary with data split by year
    """
    # Ensure datetime column is proper type
    if data.schema['datetime'] != pl.Datetime:
        data = data.with_columns(
            pl.col('datetime').str.strptime(pl.Datetime)
        )

    # Extract year and split
    data_with_year = data.with_columns(
        pl.col('datetime').dt.year().alias('year')
    )

    # Split by year
    splits = {}
    for year in [2023, 2024]:
        year_data = data_with_year.filter(pl.col('year') == year).drop('year')
        splits[str(year)] = year_data

    return splits

def create_micro_test_windows(data: pl.DataFrame, window_days: int, max_windows: int = None) -> List[pl.DataFrame]:
    """
    Create micro-period test windows from 2024 data

    Args:
        data: 2024 test data
        window_days: Size of each micro window in days
        max_windows: Maximum windows to create

    Returns:
        List of micro-period windows
    """
    if len(data) == 0:
        return []

    # Sort by datetime
    data_sorted = data.sort('datetime')

    start_date = data_sorted['datetime'].min()
    end_date = data_sorted['datetime'].max()

    # Handle datetime conversion if needed
    if hasattr(start_date, 'to_py'):
        start_date = start_date.to_py()
        end_date = end_date.to_py()

    total_duration = (end_date - start_date).total_seconds() / 86400  # days

    if total_duration <= window_days:
        return [data_sorted]

    # Use daily stride for comprehensive coverage
    stride_days = 1
    if max_windows is not None:
        stride_days = max(1, int((total_duration - window_days) / max(max_windows - 1, 1)))

    windows = []
    current_start = start_date
    window_count = 0

    while current_start < end_date and (max_windows is None or window_count < max_windows):
        window_end = current_start + timedelta(days=window_days)

        if window_end > end_date:
            window_end = end_date

        # Filter data for this window
        window_data = data_sorted.filter(
            (pl.col('datetime') >= current_start) &
            (pl.col('datetime') <= window_end)
        )

        # Only add windows with sufficient data for SFM processing
        if len(window_data) >= 50:
            windows.append(window_data)
            window_count += 1

        current_start += timedelta(days=stride_days)

    return windows

def train_on_year(train_data: pl.DataFrame, sfm_model) -> Any:
    """
    Train SFM model on specific year data

    Args:
        train_data: Training data (2023)
        sfm_model: SFM model function

    Returns:
        Trained UEL instance
    """
    print(f"Training on {len(train_data)} rows from 2023...")
    print(f"Date range: {train_data['datetime'].min()} to {train_data['datetime'].max()}")

    # Create UEL instance for training
    uel = loop.UniversalExperimentLoop(
        data=train_data,
        single_file_model=sfm_model
    )

    # Train model
    uel.run(
        experiment_name='train_2023',
        n_permutations=1,
        prep_each_round=True
    )

    if len(uel.experiment_log) == 0:
        raise ValueError("Failed to train model on 2023 data")

    result = uel.experiment_log.row(0, named=True)
    print(f"Training complete. AUC: {result.get('auc', 'N/A'):.4f}")

    return uel

def test_on_micro_window(window_data: pl.DataFrame, sfm_model, verbose: bool = False) -> Dict[str, float]:
    """
    Test model on micro-period window (independent training)

    Args:
        window_data: Test window data
        sfm_model: SFM model function
        verbose: Show progress

    Returns:
        Performance metrics
    """
    try:
        if verbose:
            min_date = window_data['datetime'].min()
            max_date = window_data['datetime'].max()

            # Handle datetime conversion if needed
            if hasattr(min_date, 'to_py'):
                min_date = min_date.to_py()
                max_date = max_date.to_py()

            print(f"    Testing on {len(window_data)} rows ({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})...")

        # Create fresh UEL instance for this micro-period test
        test_uel = loop.UniversalExperimentLoop(
            data=window_data,
            single_file_model=sfm_model
        )

        # Train and test on this micro-period
        test_uel.run(
            experiment_name=f'micro_test_{len(window_data)}_rows',
            n_permutations=1,
            prep_each_round=True
        )

        if len(test_uel.experiment_log) > 0:
            result = test_uel.experiment_log.row(0, named=True)

            metrics = {
                'auc': result.get('auc', np.nan),
                'precision': result.get('precision', np.nan),
                'recall': result.get('recall', np.nan),
                'f1': result.get('f1', np.nan),
                'trading_return_net_pct': result.get('trading_return_net_pct', np.nan),
                'trading_win_rate_pct': result.get('trading_win_rate_pct', np.nan),
                'win_loss_ratio': abs(result.get('trading_avg_win', np.nan) / result.get('trading_avg_loss', 1)) if result.get('trading_avg_loss', 0) != 0 else np.nan
            }

            if verbose and not np.isnan(metrics['auc']):
                print(f"    AUC: {metrics['auc']:.4f}, Return: {metrics['trading_return_net_pct']:.4f}%")

            return metrics
        else:
            if verbose:
                print("    No results obtained")
            return {k: np.nan for k in ['auc', 'precision', 'recall', 'f1', 'trading_return_net_pct', 'trading_win_rate_pct', 'win_loss_ratio']}

    except Exception as e:
        if verbose:
            print(f"    Error: {str(e)[:60]}...")
        return {k: np.nan for k in ['auc', 'precision', 'recall', 'f1', 'trading_return_net_pct', 'trading_win_rate_pct', 'win_loss_ratio']}

def calculate_stability_metrics(values: List[float]) -> Dict[str, float]:
    """Calculate stability metrics for performance values"""
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

def proper_train_test_analysis(data: pl.DataFrame,
                              sfm_model,
                              test_window_days: List[int] = [1, 2, 3],
                              max_test_windows: int = 50,
                              verbose: bool = False) -> Dict[str, Any]:
    """
    Proper train-test micro-period analysis

    Args:
        data: Full dataset
        sfm_model: SFM model to analyze
        test_window_days: List of micro-period sizes to test
        max_test_windows: Maximum test windows per size
        verbose: Show progress

    Returns:
        Analysis results
    """
    print("=== PROPER TRAIN-TEST MICRO-PERIOD ANALYSIS ===")
    print("Train on 2023, test on 2024 micro-periods")

    # Split data by year
    year_splits = split_data_by_year(data)

    if '2023' not in year_splits or len(year_splits['2023']) == 0:
        raise ValueError("No 2023 data available for training")
    if '2024' not in year_splits or len(year_splits['2024']) == 0:
        raise ValueError("No 2024 data available for testing")

    train_data = year_splits['2023']
    test_data = year_splits['2024']

    print(f"Train data (2023): {len(train_data)} rows")
    print(f"Test data (2024): {len(test_data)} rows")

    # Train model on 2023 data
    trained_model = train_on_year(train_data, sfm_model)

    # Test on 2024 micro-periods
    results = {}

    for window_days in test_window_days:
        print(f"\nTesting {window_days}-day micro-periods on 2024 data...")

        # Create micro-period test windows from 2024
        test_windows = create_micro_test_windows(test_data, window_days, max_test_windows)
        print(f"Created {len(test_windows)} test windows from 2024")

        if len(test_windows) == 0:
            print(f"No valid test windows for {window_days}-day analysis")
            continue

        # Test each micro-period independently
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
            if verbose or (i % 10 == 0):
                print(f"  Testing window {i}/{len(test_windows)}...")

            window_metrics = test_on_micro_window(window, sfm_model, verbose)

            # Check for valid results
            has_valid = any(not np.isnan(v) for v in window_metrics.values())
            if has_valid:
                valid_tests += 1

            # Store all results
            for metric, value in window_metrics.items():
                all_metrics[metric].append(value)

        print(f"  Completed: {len(test_windows)} windows, {valid_tests} with valid results")

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
    print("Loading dataset...")
    data = get_klines_data()
    print(f"Total dataset: {len(data)} rows")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")

    # Run proper train-test analysis
    results = proper_train_test_analysis(
        data=data,
        sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
        test_window_days=[7, 10, 14],  # Micro-periods with adequate data
        max_test_windows=50,  # Reasonable number for testing
        verbose=True
    )

    # Display results
    print("\n" + "="*80)
    print("TRAIN-TEST MICRO-PERIOD RESULTS")
    print("="*80)
    print("Trained on: 2023 data")
    print("Tested on: 2024 micro-periods")

    for window_key, window_results in results.items():
        print(f"\n{window_key.upper()} MICRO-PERIODS:")
        print(f"  Total windows tested: {window_results.get('total_windows', 0)}")
        print(f"  Windows with valid results: {window_results.get('valid_windows', 0)}")

        if window_results.get('valid_windows', 0) > 0:
            print("  Performance stability across 2024 micro-periods:")
            key_metrics = ['auc', 'trading_return_net_pct', 'win_loss_ratio']

            for metric in key_metrics:
                if metric in window_results['stability']:
                    stats = window_results['stability'][metric]
                    mean_val = stats.get('mean')
                    if mean_val is not None and not np.isnan(mean_val):
                        print(f"    {metric}: mean={mean_val:.4f}, std={stats.get('std', 0):.4f}, coeff_var={stats.get('coeff_var', 0):.4f}")
                        print(f"      range: {stats.get('percentile_5', 0):.4f} - {stats.get('percentile_95', 0):.4f}")
                    else:
                        print(f"    {metric}: No valid data")
        else:
            print("  No valid results obtained")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("This shows how model performance varies across different micro-periods")
    print("in out-of-sample 2024 data after training on 2023.")
    print(f"{'='*80}")