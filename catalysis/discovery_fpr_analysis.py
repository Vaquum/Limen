#!/usr/bin/env python3
"""
Discovery Run 3: FPR (False Positive Rate) Analysis
Analyze the relationship between confidence thresholds, FPR, and economic performance
"""

import warnings
warnings.filterwarnings('ignore')

import loop
import polars as pl
import pandas as pd
import numpy as np


def analyze_fpr_patterns():
    """
    Run tradeline_long_binary with focused confidence threshold variations to understand:
    1. Confidence threshold vs FPR relationship
    2. FPR vs economic performance trade-offs
    3. Optimal FPR ranges for cohort selection
    4. Signal frequency vs quality trade-offs
    """

    print("Loading historical data...")
    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=3600, start_date_limit='2022-01-01')  # Use recent 3+ years

    print(f"Dataset: {len(historical.data)} rows from {historical.data['datetime'].min()} to {historical.data['datetime'].max()}")

    print("\n=== Running FPR Analysis Discovery ===")
    print("Focusing on confidence threshold variations...")

    # Create focused parameter space for FPR analysis
    print("Creating FPR-focused parameter space...")

    # We'll modify the tradeline_long_binary params to focus on confidence thresholds
    # and fix other parameters to reduce noise

    try:
        # Use a custom parameter configuration for FPR analysis
        class FPRAnalysisParams:
            @staticmethod
            def params():
                return {
                    # Fixed feature engineering parameters for consistency
                    'quantile_threshold': [0.75],  # Fixed
                    'min_height_pct': [0.003],     # Fixed
                    'max_duration_hours': [48],    # Fixed
                    'lookahead_hours': [48],       # Fixed
                    'long_threshold_percentile': [75],  # Fixed

                    # Fixed trading parameters except confidence
                    'loser_timeout_hours': [24],   # Fixed
                    'max_hold_hours': [48],        # Fixed
                    'position_size': [0.20],       # Fixed
                    'min_stop_loss': [0.010],      # Fixed
                    'max_stop_loss': [0.040],      # Fixed
                    'atr_stop_multiplier': [1.5],  # Fixed
                    'trailing_activation': [0.02], # Fixed
                    'trailing_distance': [0.5],    # Fixed
                    'default_atr_pct': [0.015],    # Fixed

                    # Fixed model hyperparameters
                    'num_leaves': [63],            # Fixed
                    'learning_rate': [0.1],       # Fixed
                    'feature_fraction': [0.9],    # Fixed
                    'bagging_fraction': [0.9],    # Fixed
                    'bagging_freq': [5],          # Fixed
                    'min_child_samples': [20],    # Fixed
                    'lambda_l1': [0],             # Fixed
                    'lambda_l2': [0],             # Fixed
                    'n_estimators': [500],        # Fixed

                    # VARY ONLY CONFIDENCE THRESHOLD for FPR analysis
                    'confidence_threshold': [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
                }

            @staticmethod
            def prep(data, round_params=None):
                return loop.sfm.lightgbm.tradeline_long_binary.prep(data, round_params)

            @staticmethod
            def model(data, round_params):
                return loop.sfm.lightgbm.tradeline_long_binary.model(data, round_params)

        # Create temporary SFM object for FPR analysis
        fpr_analysis_sfm = FPRAnalysisParams()

        # Run UEL with FPR-focused parameter space
        uel = loop.UniversalExperimentLoop(
            data=historical.data,
            single_file_model=fpr_analysis_sfm
        )

        # Run experiments - we need enough samples per confidence level
        n_confidence_levels = len(fpr_analysis_sfm.params()['confidence_threshold'])
        samples_per_level = 20  # 20 samples per confidence threshold
        total_permutations = n_confidence_levels * samples_per_level

        print(f"Running {total_permutations} permutations ({samples_per_level} per confidence level)...")

        uel.run(
            experiment_name="discovery_fpr_analysis",
            n_permutations=total_permutations,
            prep_each_round=True,
            save_to_sqlite=False
        )

        # Extract metrics
        backtest_results = uel.experiment_backtest_results
        experiment_log = uel.experiment_log

        print(f"\nAnalyzing {len(backtest_results)} model permutations...")

        # Merge all metrics for analysis
        analysis_df = backtest_results.copy()
        analysis_df['fpr'] = experiment_log['fpr']
        analysis_df['precision'] = experiment_log['precision']
        analysis_df['recall'] = experiment_log['recall']
        analysis_df['f1_score'] = experiment_log['f1']

        # Add confidence threshold from experiment log
        if 'confidence_threshold' in experiment_log.columns:
            analysis_df['confidence_threshold'] = experiment_log['confidence_threshold']
        else:
            print("Warning: confidence_threshold not found in experiment_log")

        print("\n=== FPR ANALYSIS BY CONFIDENCE THRESHOLD ===")

        # Group by confidence threshold
        confidence_levels = sorted(analysis_df['confidence_threshold'].unique())

        fpr_summary = []
        for conf_threshold in confidence_levels:
            subset = analysis_df[analysis_df['confidence_threshold'] == conf_threshold]

            summary = {
                'confidence_threshold': conf_threshold,
                'sample_count': len(subset),
                'avg_fpr': subset['fpr'].mean(),
                'std_fpr': subset['fpr'].std(),
                'median_fpr': subset['fpr'].median(),
                'min_fpr': subset['fpr'].min(),
                'max_fpr': subset['fpr'].max(),
                'avg_return_net_pct': subset['total_return_net_pct'].mean(),
                'std_return_net_pct': subset['total_return_net_pct'].std(),
                'profitable_models': (subset['total_return_net_pct'] > 0).sum(),
                'profitable_ratio': (subset['total_return_net_pct'] > 0).mean(),
                'avg_trades_count': subset['trades_count'].mean(),
                'avg_win_rate': subset['trade_win_rate_pct'].mean(),
                'avg_sharpe': subset['sharpe_per_bar'].mean(),
                'avg_precision': subset['precision'].mean(),
                'avg_recall': subset['recall'].mean(),
                'avg_f1_score': subset['f1_score'].mean()
            }
            fpr_summary.append(summary)

            print(f"\nConfidence {conf_threshold:.2f}:")
            print(f"  FPR: {summary['avg_fpr']:.3f} ± {summary['std_fpr']:.3f}")
            print(f"  Return: {summary['avg_return_net_pct']:.2f}% ± {summary['std_return_net_pct']:.2f}%")
            print(f"  Profitable: {summary['profitable_models']}/{summary['sample_count']} ({summary['profitable_ratio']:.1%})")
            print(f"  Trades: {summary['avg_trades_count']:.1f}")
            print(f"  Precision: {summary['avg_precision']:.3f}")

        fpr_summary_df = pd.DataFrame(fpr_summary)

        print("\n=== FPR RECOMMENDATIONS ===")

        # Find optimal FPR ranges
        viable_summaries = fpr_summary_df[
            (fpr_summary_df['avg_return_net_pct'] > 0) &
            (fpr_summary_df['profitable_ratio'] > 0.4) &
            (fpr_summary_df['avg_trades_count'] >= 5)
        ]

        if len(viable_summaries) > 0:
            best_fpr_range = {
                'min_fpr': viable_summaries['avg_fpr'].min(),
                'max_fpr': viable_summaries['avg_fpr'].max(),
                'optimal_fpr': viable_summaries.loc[viable_summaries['avg_return_net_pct'].idxmax(), 'avg_fpr'],
                'optimal_confidence': viable_summaries.loc[viable_summaries['avg_return_net_pct'].idxmax(), 'confidence_threshold']
            }

            print(f"Viable FPR range: {best_fpr_range['min_fpr']:.3f} - {best_fpr_range['max_fpr']:.3f}")
            print(f"Optimal FPR: {best_fpr_range['optimal_fpr']:.3f} (confidence: {best_fpr_range['optimal_confidence']:.2f})")

            # Recommend confidence threshold ranges for cohort
            viable_confidences = viable_summaries['confidence_threshold'].tolist()
            print(f"Recommended confidence thresholds for cohort: {viable_confidences}")

        else:
            best_fpr_range = {
                'min_fpr': fpr_summary_df['avg_fpr'].quantile(0.25),
                'max_fpr': fpr_summary_df['avg_fpr'].quantile(0.50),
                'optimal_fpr': fpr_summary_df.loc[fpr_summary_df['avg_return_net_pct'].idxmax(), 'avg_fpr'],
                'optimal_confidence': fpr_summary_df.loc[fpr_summary_df['avg_return_net_pct'].idxmax(), 'confidence_threshold']
            }
            print("No clearly viable models found, using statistical approach:")
            print(f"Conservative FPR range: {best_fpr_range['min_fpr']:.3f} - {best_fpr_range['max_fpr']:.3f}")

        # Save results
        analysis_df.to_csv('catalysis/fpr_detailed_analysis.csv', index=False)
        fpr_summary_df.to_csv('catalysis/fpr_summary_by_confidence.csv', index=False)

        # Save FPR recommendations
        fpr_recommendations = pd.DataFrame([best_fpr_range])
        fpr_recommendations.to_csv('catalysis/fpr_recommendations.csv', index=False)

        print(f"\nResults saved to:")
        print(f"  catalysis/fpr_detailed_analysis.csv")
        print(f"  catalysis/fpr_summary_by_confidence.csv")
        print(f"  catalysis/fpr_recommendations.csv")

        return analysis_df, fpr_summary_df, best_fpr_range

    except Exception as e:
        print(f"Error in FPR analysis: {e}")
        raise


if __name__ == "__main__":
    analysis_df, summary_df, recommendations = analyze_fpr_patterns()