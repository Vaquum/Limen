#!/usr/bin/env python3
"""
Full-Scale Micro-Period Sensitivity Analysis Runner
Designed to process all available windows without artificial limits
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import polars as pl
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import loop
from loop.tests.utils.get_data import get_klines_data
from sfm_micro_period_analysis import micro_period_sensitivity_analysis
from sfm_sensitivity_runner import generate_parameter_combinations

def run_full_scale_analysis(n_permutations: int = 10,
                          window_days_list: List[int] = [1, 2, 3, 5, 10]) -> Dict[str, Any]:
    """
    Run full-scale sensitivity analysis with NO window limits

    Args:
        n_permutations: Number of parameter combinations to test
        window_days_list: Window sizes to analyze

    Returns:
        Comprehensive results dictionary
    """
    print("="*80)
    print("FULL-SCALE MICRO-PERIOD SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Target: Process ALL available windows (no artificial limits)")
    print(f"Permutations: {n_permutations}")
    print(f"Window sizes: {window_days_list}")
    print("")

    # Load full dataset
    print("Loading FULL dataset...")
    data = get_klines_data()
    print(f"Dataset size: {len(data)} rows")

    # Estimate total windows for each window size
    start_date = data['datetime'].min()
    end_date = data['datetime'].max()
    total_duration_days = (end_date - start_date).total_seconds() / 86400

    print(f"Date range: {start_date} to {end_date}")
    print(f"Total duration: {total_duration_days:.1f} days")
    print("")

    # Estimate windows per size
    estimated_windows = {}
    for window_days in window_days_list:
        est_count = max(1, int(total_duration_days - window_days + 1))
        estimated_windows[window_days] = est_count
        print(f"  {window_days}-day windows: ~{est_count} windows expected")

    total_estimated = sum(estimated_windows.values()) * n_permutations
    print(f"\\nTotal estimated window-permutation combinations: {total_estimated}")
    print("")

    # Generate parameter combinations
    print(f"Generating {n_permutations} diverse parameter combinations...")
    param_combinations = generate_parameter_combinations(n_permutations)

    all_results = {}

    for i, params in enumerate(param_combinations):
        print(f"\\n{'='*80}")
        print(f"PERMUTATION {i+1}/{n_permutations}")
        print(f"{'='*80}")

        # Show key parameters
        key_params = ['learning_rate', 'num_leaves', 'confidence_threshold',
                     'position_size', 'max_duration_hours']
        param_summary = {k: params.get(k, 'N/A') for k in key_params}
        print(f"Key parameters: {param_summary}")

        try:
            # Run micro-period analysis with NO window limits
            perm_results = micro_period_sensitivity_analysis(
                data=data,
                sfm_model=loop.sfm.lightgbm.tradeline_long_binary,
                window_days_list=window_days_list,
                specific_params=params,
                max_windows=None,  # NO LIMIT - process all windows
                verbose=True
            )

            # Store results
            all_results[f"permutation_{i+1}"] = {
                'parameters': params,
                'results': perm_results
            }

            # Show summary for this permutation
            print(f"\\nPermutation {i+1} summary:")
            for window_size, window_results in perm_results.items():
                valid_windows = window_results['valid_windows']
                total_windows = window_results['total_windows']
                success_rate = (valid_windows / total_windows * 100) if total_windows > 0 else 0

                print(f"  {window_size}: {valid_windows}/{total_windows} valid windows ({success_rate:.1f}% success)")

                # Show AUC stats if available
                auc_stability = window_results['stability']['auc']
                if not np.isnan(auc_stability['mean']):
                    print(f"    AUC: mean={auc_stability['mean']:.4f}, std={auc_stability['std']:.4f}")

        except Exception as e:
            print(f"\\n❌ ERROR in permutation {i+1}: {str(e)}")
            print("Continuing with next permutation...")
            all_results[f"permutation_{i+1}"] = {
                'parameters': params,
                'results': {},
                'error': str(e)
            }

    return all_results

def save_comprehensive_results(results: Dict[str, Any],
                             window_days_list: List[int],
                             suffix: str = "full_scale") -> Tuple[str, str]:
    """Save results and generate comprehensive report"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        else:
            return convert_numpy(obj)

    # Save raw results
    results_file = f"/Users/beyondsyntax/Loop/catalysis/sensitivity_results_{suffix}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(deep_convert(results), f, indent=2)

    # Generate comprehensive report
    report = []
    report.append("=" * 100)
    report.append("FULL-SCALE MICRO-PERIOD SENSITIVITY ANALYSIS REPORT")
    report.append("=" * 100)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Analysis scope: ALL available windows (no artificial limits)")
    report.append(f"Permutations tested: {len(results)}")
    report.append(f"Window sizes: {window_days_list}")
    report.append("")

    # Summary statistics
    total_windows_all_perms = 0
    total_valid_all_perms = 0

    for window_days in window_days_list:
        window_key = f"{window_days}d"
        report.append(f"\\n{'='*80}")
        report.append(f"{window_days}-DAY WINDOW ANALYSIS")
        report.append(f"{'='*80}")

        # Collect data for this window size across all permutations
        window_data = []
        for perm_name, perm_data in results.items():
            if 'results' in perm_data and window_key in perm_data['results']:
                window_results = perm_data['results'][window_key]
                total_windows = window_results['total_windows']
                valid_windows = window_results['valid_windows']

                total_windows_all_perms += total_windows
                total_valid_all_perms += valid_windows

                # Get AUC stability metrics
                auc_stability = window_results['stability']['auc']

                window_data.append({
                    'permutation': perm_name,
                    'total_windows': total_windows,
                    'valid_windows': valid_windows,
                    'success_rate': (valid_windows / total_windows * 100) if total_windows > 0 else 0,
                    'auc_mean': auc_stability['mean'],
                    'auc_std': auc_stability['std'],
                    'auc_coeff_var': auc_stability['coeff_var']
                })

        if window_data:
            # Sort by success rate
            window_data.sort(key=lambda x: x['success_rate'], reverse=True)

            report.append(f"Total permutations: {len(window_data)}")
            report.append(f"Average windows per permutation: {np.mean([w['total_windows'] for w in window_data]):.1f}")
            report.append(f"Average success rate: {np.mean([w['success_rate'] for w in window_data]):.1f}%")
            report.append("")

            report.append("Top performing permutations (by success rate):")
            for i, entry in enumerate(window_data[:5]):
                report.append(f"  {i+1}. {entry['permutation']}:")
                report.append(f"     Windows: {entry['valid_windows']}/{entry['total_windows']} ({entry['success_rate']:.1f}%)")
                if not np.isnan(entry['auc_mean']):
                    report.append(f"     AUC: {entry['auc_mean']:.4f} ± {entry['auc_std']:.4f}")
                report.append("")
        else:
            report.append("No valid data for this window size")

    # Overall summary
    report.append(f"\\n{'='*80}")
    report.append("OVERALL ANALYSIS SUMMARY")
    report.append(f"{'='*80}")
    report.append(f"Total window-permutation combinations processed: {total_windows_all_perms}")
    report.append(f"Total combinations with valid results: {total_valid_all_perms}")

    if total_windows_all_perms > 0:
        overall_success_rate = (total_valid_all_perms / total_windows_all_perms * 100)
        report.append(f"Overall success rate: {overall_success_rate:.1f}%")

    report_text = "\\n".join(report)

    # Save report
    report_file = f"/Users/beyondsyntax/Loop/catalysis/sensitivity_report_{suffix}_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)

    return results_file, report_file

if __name__ == "__main__":
    print("Starting FULL-SCALE micro-period sensitivity analysis...")
    print("WARNING: This will process ALL available windows - may take significant time!")
    print("")

    # Run the full analysis
    results = run_full_scale_analysis(
        n_permutations=5,  # Start with 5 permutations for testing
        window_days_list=[1, 2, 3]  # Small window sizes to maximize coverage
    )

    # Save results
    results_file, report_file = save_comprehensive_results(
        results,
        window_days_list=[1, 2, 3],
        suffix="full_scale"
    )

    print(f"\\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to:")
    print(f"  Raw data: {results_file}")
    print(f"  Report: {report_file}")

    # Quick summary
    print(f"\\nQuick Summary:")
    total_permutations = len(results)
    successful_permutations = 0

    for perm_name, perm_data in results.items():
        if 'results' in perm_data and perm_data['results']:
            has_valid_results = False
            for window_size, window_results in perm_data['results'].items():
                if window_results['valid_windows'] > 0:
                    has_valid_results = True
                    break
            if has_valid_results:
                successful_permutations += 1

    print(f"  Permutations processed: {total_permutations}")
    print(f"  Permutations with valid results: {successful_permutations}")
    print(f"  Success rate: {(successful_permutations/total_permutations*100):.1f}%")