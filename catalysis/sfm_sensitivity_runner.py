#!/usr/bin/env python3
"""
SFM Multi-Permutation Micro-Period Sensitivity Analysis Runner
Runs multiple model permutations and compares their temporal stability
"""

import pandas as pd
import polars as pl
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import loop
from loop.tests.utils.get_data import get_klines_data
from sfm_micro_period_analysis import micro_period_sensitivity_analysis


def generate_parameter_combinations(n_combinations: int = 10) -> List[Dict[str, Any]]:
    """
    Generate diverse parameter combinations for testing

    Args:
        n_combinations: Number of parameter sets to generate

    Returns:
        List of parameter dictionaries
    """
    np.random.seed(42)  # For reproducible results

    combinations = []

    # Base parameter ranges from tradeline_long_binary
    param_ranges = {
        'quantile_threshold': [0.60, 0.70, 0.75, 0.80, 0.85],
        'min_height_pct': [0.001, 0.002, 0.003, 0.004, 0.005],
        'max_duration_hours': [24, 48, 72, 96],
        'lookahead_hours': [24, 48, 72],
        'long_threshold_percentile': [65, 75, 85],
        'confidence_threshold': [0.40, 0.50, 0.60],
        'position_size': [0.10, 0.20, 0.30],
        'min_stop_loss': [0.005, 0.010, 0.020],
        'max_stop_loss': [0.030, 0.040, 0.050],
        'atr_stop_multiplier': [1.0, 1.5, 2.0],
        'trailing_activation': [0.01, 0.02, 0.03],
        'trailing_distance': [0.3, 0.5, 0.7],
        'loser_timeout_hours': [12, 24, 48],
        'max_hold_hours': [24, 48, 72, 96],
        'default_atr_pct': [0.010, 0.015, 0.020],
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.05, 0.1],
        'feature_fraction': [0.9],
        'bagging_fraction': [0.8, 0.9],
        'bagging_freq': [5],
        'min_child_samples': [10, 20],
        'lambda_l1': [0, 0.1],
        'lambda_l2': [0, 0.1],
        'n_estimators': [500]
    }

    for i in range(n_combinations):
        params = {}
        for param, values in param_ranges.items():
            params[param] = np.random.choice(values)
        combinations.append(params)

    return combinations


def run_multi_permutation_analysis(data: pl.DataFrame,
                                 n_permutations: int = 5,
                                 window_days_list: List[int] = [30, 60, 90],
                                 max_windows_per_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Run sensitivity analysis across multiple model permutations

    Args:
        data: Full dataset
        n_permutations: Number of different parameter combinations to test
        window_days_list: List of window sizes to test
        max_windows_per_size: Maximum windows per window size (None = no limit)

    Returns:
        Comprehensive results dictionary
    """
    print(f"Generating {n_permutations} parameter combinations...")
    param_combinations = generate_parameter_combinations(n_permutations)

    all_results = {}

    for i, params in enumerate(param_combinations):
        print(f"\\n{'='*60}")
        print(f"RUNNING PERMUTATION {i+1}/{n_permutations}")
        print(f"{'='*60}")

        # Run micro-period analysis with specific parameters
        print(f"Using parameters: {params}")
        perm_results = micro_period_sensitivity_analysis(
            data=data,
            sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
            window_days_list=window_days_list,
            specific_params=params,
            max_windows=max_windows_per_size,  # Use configurable limit
            verbose=True  # Show progress
        )

        all_results[f"permutation_{i+1}"] = {
            'parameters': params,
            'results': perm_results
        }

    return all_results


def create_stability_ranking(results: Dict[str, Any],
                           metric: str = 'auc',
                           window_size: str = '1d') -> List[Dict[str, Any]]:
    """
    Rank permutations by stability of a specific metric

    Args:
        results: Results from run_multi_permutation_analysis
        metric: Which metric to rank by
        window_size: Which window size to analyze

    Returns:
        List of permutations ranked by stability (lowest coefficient of variation first)
    """
    ranking = []

    for perm_name, perm_data in results.items():
        if window_size in perm_data['results']:
            stability = perm_data['results'][window_size]['stability'][metric]

            ranking.append({
                'permutation': perm_name,
                'parameters': perm_data['parameters'],
                'mean': stability['mean'],
                'std': stability['std'],
                'coeff_var': stability['coeff_var'],
                'range': stability['range'],
                'valid_windows': stability['valid_windows']
            })

    # Sort by coefficient of variation (lower = more stable)
    ranking.sort(key=lambda x: x['coeff_var'] if not np.isnan(x['coeff_var']) else float('inf'))

    return ranking


def generate_comprehensive_report(results: Dict[str, Any],
                                window_days_list: List[int] = [1, 2, 3]) -> str:
    """
    Generate a comprehensive text report of sensitivity analysis results

    Args:
        results: Results from run_multi_permutation_analysis
        window_days_list: Window sizes that were tested

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("SFM MICRO-PERIOD SENSITIVITY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Permutations tested: {len(results)}")
    report.append(f"Window sizes: {window_days_list}")
    report.append("")

    # Key metrics to focus on
    key_metrics = ['auc', 'trading_return_net_pct', 'win_loss_ratio']

    for window_days in window_days_list:
        window_key = f"{window_days}d"
        report.append(f"\\n{'='*60}")
        report.append(f"{window_days}-DAY WINDOW ANALYSIS")
        report.append(f"{'='*60}")

        for metric in key_metrics:
            report.append(f"\\n--- {metric.upper()} STABILITY RANKING ---")
            ranking = create_stability_ranking(results, metric, window_key)

            for i, entry in enumerate(ranking[:5]):  # Top 5 most stable
                report.append(f"{i+1}. {entry['permutation']}:")
                report.append(f"   Mean: {entry['mean']:.4f}")
                report.append(f"   Std: {entry['std']:.4f}")
                report.append(f"   CoeffVar: {entry['coeff_var']:.4f}")
                report.append(f"   Range: {entry['range']:.4f}")
                report.append(f"   Valid Windows: {entry['valid_windows']}")

                # Show key parameters
                key_params = ['learning_rate', 'num_leaves', 'confidence_threshold', 'position_size']
                param_str = ", ".join([f"{p}: {entry['parameters'].get(p, 'N/A')}"
                                     for p in key_params])
                report.append(f"   Key Params: {param_str}")
                report.append("")

    # Cross-window stability analysis
    report.append(f"\\n{'='*60}")
    report.append("CROSS-WINDOW STABILITY COMPARISON")
    report.append(f"{'='*60}")

    for metric in key_metrics:
        report.append(f"\\n--- {metric.upper()} ACROSS WINDOW SIZES ---")

        # Find permutations that are consistently stable across window sizes
        stability_scores = {}

        for perm_name, perm_data in results.items():
            scores = []
            for window_days in window_days_list:
                window_key = f"{window_days}d"
                if window_key in perm_data['results']:
                    coeff_var = perm_data['results'][window_key]['stability'][metric]['coeff_var']
                    if not np.isnan(coeff_var):
                        scores.append(coeff_var)

            if scores:
                stability_scores[perm_name] = np.mean(scores)

        # Rank by average stability across window sizes
        sorted_perms = sorted(stability_scores.items(), key=lambda x: x[1])

        for i, (perm_name, avg_coeff_var) in enumerate(sorted_perms[:3]):
            report.append(f"{i+1}. {perm_name}: Avg CoeffVar = {avg_coeff_var:.4f}")

    return "\\n".join(report)


if __name__ == "__main__":
    print("Loading test data...")
    from loop.tests.utils.get_data import get_klines_data_small

    # Use small dataset for faster testing
    data = get_klines_data_small()

    # Further limit to 800 rows for quick testing
    test_data = data.head(800)

    print(f"Using test dataset with {len(test_data)} rows for quick analysis...")

    print("Running optimized multi-permutation micro-period sensitivity analysis...")
    results = run_multi_permutation_analysis(
        data=test_data,
        n_permutations=3,  # Start small for testing
        window_days_list=[5, 10],  # Use smaller windows for quick testing
        max_windows_per_size=6  # Limit for quick testing
    )

    # Generate comprehensive report
    report = generate_comprehensive_report(results, [5, 10])

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save raw results as JSON
    results_file = f"/Users/beyondsyntax/Loop/catalysis/sensitivity_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Deep convert all numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)

        json.dump(deep_convert(results), f, indent=2)

    # Save report
    report_file = f"/Users/beyondsyntax/Loop/catalysis/sensitivity_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\\nResults saved to:")
    print(f"  Raw data: {results_file}")
    print(f"  Report: {report_file}")

    print("\\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print(report[:2000] + "\\n[Report truncated - see full report file]")