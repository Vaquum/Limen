#!/usr/bin/env python3
"""
SFM Micro-Period Sensitivity Analysis
Evaluates how SFM models perform across different time windows
"""

import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import loop
from loop.tests.utils.get_data import get_klines_data


def create_time_windows(data: pl.DataFrame, window_days: int, max_windows: int = None) -> List[pl.DataFrame]:
    """
    Split data into rolling time windows of specified length with intelligent sampling

    Args:
        data: Polars DataFrame with datetime column
        window_days: Length of each window in days
        max_windows: Maximum number of windows to create (default: None for no limit)

    Returns:
        List of DataFrames, each representing one time window
    """
    if 'datetime' not in data.columns:
        raise ValueError("Data must have a 'datetime' column")

    # Sort by datetime
    data_sorted = data.sort('datetime')

    # Convert to datetime if not already
    if data_sorted.schema['datetime'] != pl.Datetime:
        data_sorted = data_sorted.with_columns(
            pl.col('datetime').str.strptime(pl.Datetime)
        )

    # Get date range
    start_date = data_sorted['datetime'].min()
    end_date = data_sorted['datetime'].max()

    # Handle datetime difference calculation properly
    if hasattr(start_date, 'total_seconds'):
        # Standard datetime objects
        total_duration = (end_date - start_date).total_seconds() / 86400  # days
    else:
        # Polars datetime - convert to python datetime first
        start_py = start_date.to_py() if hasattr(start_date, 'to_py') else start_date
        end_py = end_date.to_py() if hasattr(end_date, 'to_py') else end_date
        total_duration = (end_py - start_py).total_seconds() / 86400  # days

    # Calculate stride to get approximately max_windows windows
    if total_duration <= window_days:
        # If total data is less than window size, create one window with all data
        return [data_sorted]

    # If no max_windows limit, use daily stride for comprehensive analysis
    if max_windows is None:
        stride_days = 1  # Daily stride for maximum coverage
    else:
        stride_days = max(1, int((total_duration - window_days) / max(max_windows - 1, 1)))

    windows = []
    current_start = start_date
    window_count = 0

    while current_start < end_date and (max_windows is None or window_count < max_windows):
        window_end = current_start + timedelta(days=window_days)

        if window_end > end_date:
            # Last window: extend to include all remaining data
            window_end = end_date

        # Filter data for this window
        window_data = data_sorted.filter(
            (pl.col('datetime') >= current_start) &
            (pl.col('datetime') <= window_end)
        )

        # Only add windows with sufficient data (at least 10 rows for micro-period analysis)
        if len(window_data) >= 10:
            windows.append(window_data)
            window_count += 1

        # Move to next window
        current_start += timedelta(days=stride_days)

    return windows


def extract_key_metrics(uel_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract the 8 key metrics from UEL results

    Args:
        uel_results: Dictionary containing UEL experiment results

    Returns:
        Dictionary with 8 key metrics
    """
    metrics = {}

    # Model Performance
    metrics['auc'] = uel_results.get('auc', np.nan)
    metrics['val_loss'] = uel_results.get('val_loss', np.nan)

    # Prediction Quality
    metrics['precision'] = uel_results.get('precision', np.nan)
    metrics['recall'] = uel_results.get('recall', np.nan)
    metrics['f1'] = uel_results.get('f1', np.nan)

    # Trading Performance
    metrics['trading_return_net_pct'] = uel_results.get('trading_return_net_pct', np.nan)
    metrics['trading_win_rate_pct'] = uel_results.get('trading_win_rate_pct', np.nan)

    # Calculate win/loss ratio
    avg_win = uel_results.get('trading_avg_win', np.nan)
    avg_loss = uel_results.get('trading_avg_loss', np.nan)
    if not np.isnan(avg_win) and not np.isnan(avg_loss) and avg_loss != 0:
        metrics['win_loss_ratio'] = abs(avg_win / avg_loss)
    else:
        metrics['win_loss_ratio'] = np.nan

    return metrics


def run_sfm_on_window(window_data: pl.DataFrame,
                      sfm_model,
                      specific_params: Optional[Dict[str, Any]] = None,
                      verbose: bool = False) -> Dict[str, float]:
    """
    Run SFM on a single time window and extract key metrics

    Args:
        window_data: Data for this time window
        sfm_model: SFM model to use
        specific_params: If provided, use these specific parameters instead of random sampling
        verbose: If True, show progress information

    Returns:
        Dictionary with key metrics for this window
    """
    try:
        if verbose:
            print(f"    Running SFM on window with {len(window_data)} rows...")

        # Create UEL instance for this window
        uel = loop.UniversalExperimentLoop(
            data=window_data,
            single_file_model=sfm_model
        )

        # If specific parameters are provided, create a custom params function
        if specific_params is not None:
            # Create a proper ParamSpace-compatible function
            def custom_params():
                from loop.exp.param_space import ParamSpace
                # Convert params to proper ParamSpace format
                param_dict = {}
                for k, v in specific_params.items():
                    if isinstance(v, list):
                        param_dict[k] = v  # Already a list
                    else:
                        param_dict[k] = [v]  # Convert single value to list
                return param_dict

            # Run experiment with specific parameters
            uel.run(
                experiment_name=f'micro_period_window_{len(window_data)}',
                n_permutations=1,
                prep_each_round=True,
                params=custom_params
            )
        else:
            # Run experiment with 1 permutation (UEL will randomly select params)
            uel.run(
                experiment_name=f'micro_period_window_{len(window_data)}',
                n_permutations=1,
                prep_each_round=True
            )

        # Extract results from the single row
        if len(uel.experiment_log) > 0:
            result_row = uel.experiment_log.row(0, named=True)
            metrics = extract_key_metrics(result_row)
            if verbose:
                auc_val = metrics.get('auc', np.nan)
                if not np.isnan(auc_val):
                    print(f"    Results: AUC={auc_val:.4f}")
                else:
                    print(f"    Results: AUC=N/A")
            return metrics
        else:
            if verbose:
                print("    No results obtained from UEL")
            return {metric: np.nan for metric in ['auc', 'val_loss', 'precision', 'recall',
                                                'f1', 'trading_return_net_pct', 'trading_win_rate_pct',
                                                'win_loss_ratio']}

    except Exception as e:
        if verbose:
            print(f"    Error in window analysis: {str(e)[:100]}...")
        return {metric: np.nan for metric in ['auc', 'val_loss', 'precision', 'recall',
                                            'f1', 'trading_return_net_pct', 'trading_win_rate_pct',
                                            'win_loss_ratio']}


def calculate_stability_metrics(metric_values: List[float]) -> Dict[str, float]:
    """
    Calculate stability metrics for a series of values

    Args:
        metric_values: List of metric values across time windows

    Returns:
        Dictionary with stability measurements
    """
    values = np.array([v for v in metric_values if not np.isnan(v)])

    if len(values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'coeff_var': np.nan,
            'percentile_5': np.nan,
            'percentile_95': np.nan,
            'range': np.nan,
            'valid_windows': 0
        }

    mean_val = np.mean(values)
    std_val = np.std(values)

    return {
        'mean': mean_val,
        'std': std_val,
        'coeff_var': std_val / abs(mean_val) if mean_val != 0 else np.inf,
        'percentile_5': np.percentile(values, 5),
        'percentile_95': np.percentile(values, 95),
        'range': np.max(values) - np.min(values),
        'valid_windows': len(values)
    }


def micro_period_sensitivity_analysis(data: pl.DataFrame,
                                    sfm_model,
                                    window_days_list: List[int] = [5, 10, 15],
                                    specific_params: Optional[Dict[str, Any]] = None,
                                    max_windows: Optional[int] = None,
                                    verbose: bool = False) -> Dict[str, Dict]:
    """
    Run complete micro-period sensitivity analysis

    Args:
        data: Full dataset
        sfm_model: SFM model to test
        window_days_list: List of window sizes to test (default: [5, 10, 15])
        specific_params: If provided, use these specific parameters instead of random sampling
        max_windows: Maximum number of windows to create per window size (default: None for no limit)
        verbose: If True, show progress information

    Returns:
        Dictionary with results for each window size
    """
    results = {}

    for window_days in window_days_list:
        print(f"\\nAnalyzing {window_days}-day windows...")

        # Create time windows with intelligent sampling
        windows = create_time_windows(data, window_days, max_windows=max_windows)
        max_desc = f"max: {max_windows}" if max_windows is not None else "no limit"
        print(f"Created {len(windows)} windows of {window_days} days each ({max_desc})")

        if len(windows) == 0:
            print(f"  No valid windows created for {window_days}-day analysis")
            continue

        # Initialize metric storage
        all_metrics = {metric: [] for metric in ['auc', 'val_loss', 'precision', 'recall',
                                               'f1', 'trading_return_net_pct', 'trading_win_rate_pct',
                                               'win_loss_ratio']}

        # Run SFM on each window
        windows_processed = 0
        windows_with_data = 0

        for i, window in enumerate(windows):
            if len(window) < 10:  # Skip windows with too little data for micro-period analysis
                if verbose:
                    print(f"  Skipping window {i+1} - insufficient data ({len(window)} rows)")
                continue

            windows_processed += 1
            print(f"  Processing window {windows_processed}/{len(windows)} ({len(window)} rows)...")

            try:
                window_metrics = run_sfm_on_window(window, sfm_model, specific_params, verbose=verbose)
            except Exception as e:
                print(f"    ERROR in window {windows_processed}: {str(e)[:100]}...")
                window_metrics = {metric: np.nan for metric in ['auc', 'val_loss', 'precision', 'recall',
                                                            'f1', 'trading_return_net_pct', 'trading_win_rate_pct',
                                                            'win_loss_ratio']}

            # Check if we got valid metrics
            has_valid = any(not np.isnan(v) for v in window_metrics.values())
            if has_valid:
                windows_with_data += 1
                if verbose:
                    print(f"    ✓ Valid metrics obtained")
            elif verbose:
                print(f"    ✗ No valid metrics")

            # Store metrics
            for metric, value in window_metrics.items():
                all_metrics[metric].append(value)

        print(f"  Processed {windows_processed} windows, {windows_with_data} had valid metrics")

        # Calculate stability for each metric
        stability_results = {}
        for metric, values in all_metrics.items():
            stability_results[metric] = calculate_stability_metrics(values)

        results[f"{window_days}d"] = {
            'raw_metrics': all_metrics,
            'stability': stability_results,
            'total_windows': len(windows),
            'valid_windows': len([v for v in all_metrics['auc'] if not np.isnan(v)])
        }

    return results


if __name__ == "__main__":
    # Example usage with optimized settings for quick testing
    print("Loading test data...")
    from loop.tests.utils.get_data import get_klines_data_small

    # Use small dataset for faster testing (5000 rows instead of full dataset)
    data = get_klines_data_small()

    # Further limit to 1000 rows for very quick testing
    test_data = data.head(1000)

    print(f"Using test dataset with {len(test_data)} rows for quick analysis...")

    print("Running optimized micro-period sensitivity analysis...")
    results = micro_period_sensitivity_analysis(
        data=test_data,
        sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
        window_days_list=[5, 10],  # Smaller windows for quick testing
        max_windows=6,  # Maximum 6 windows per window size for quick testing
        verbose=True  # Show progress
    )

    # Print summary results
    print("\\n=== MICRO-PERIOD SENSITIVITY RESULTS ===")
    for window_size, window_results in results.items():
        print(f"\\n{window_size} Windows:")
        print(f"  Total windows: {window_results['total_windows']}")
        print(f"  Valid windows: {window_results['valid_windows']}")

        for metric in ['auc', 'trading_return_net_pct', 'win_loss_ratio']:
            stability = window_results['stability'][metric]
            print(f"  {metric}:")
            print(f"    Mean: {stability['mean']:.4f}")
            print(f"    Std: {stability['std']:.4f}")
            print(f"    CoeffVar: {stability['coeff_var']:.4f}")
            print(f"    Range: {stability['range']:.4f}")