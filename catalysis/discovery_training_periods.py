#!/usr/bin/env python3
"""
Discovery Run 1: Training Period Range Analysis
Analyze different training period lengths for tradeline_long_binary to establish optimal ranges
"""

import warnings
warnings.filterwarnings('ignore')

import loop
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def analyze_training_periods():
    """
    Run tradeline_long_binary with different training data period lengths to identify:
    1. Minimum viable training period
    2. Maximum useful training period
    3. Sweet spot ranges for cohort strategy
    """

    print("Loading historical data...")
    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=3600, start_date_limit='2020-01-01')

    full_data = historical.data
    print(f"Full dataset: {len(full_data)} rows from {full_data['datetime'].min()} to {full_data['datetime'].max()}")

    # Define training periods to test (in months)
    training_periods = [3, 6, 9, 12, 18, 24, 30, 36]

    results = []

    for period_months in training_periods:
        print(f"\n=== Testing {period_months}-month training period ===")

        # Create data subset for this training period
        # Use most recent data for consistency
        end_date = full_data['datetime'].max()
        start_date = end_date - timedelta(days=period_months * 30)  # Approximate months to days

        period_data = full_data.filter(pl.col('datetime') >= start_date)

        print(f"Period data: {len(period_data)} rows from {period_data['datetime'].min()} to {period_data['datetime'].max()}")

        if len(period_data) < 1000:  # Minimum data threshold
            print(f"Skipping {period_months}-month period - insufficient data ({len(period_data)} rows)")
            continue

        try:
            # Run UEL with this training period
            uel = loop.UniversalExperimentLoop(
                data=period_data,
                single_file_model=loop.sfm.lightgbm.tradeline_long_binary
            )

            # Use smaller n_permutations for discovery
            uel.run(
                experiment_name=f"discovery_training_{period_months}m",
                n_permutations=100,  # Small sample for discovery
                prep_each_round=True,
                save_to_sqlite=False
            )

            # Extract key metrics
            backtest_results = uel.experiment_backtest_results
            experiment_log = uel.experiment_log

            # Calculate summary statistics
            avg_return = backtest_results['total_return_net_pct'].mean()
            std_return = backtest_results['total_return_net_pct'].std()
            max_return = backtest_results['total_return_net_pct'].max()
            min_return = backtest_results['total_return_net_pct'].min()

            avg_trades = backtest_results['trades_count'].mean()
            avg_win_rate = backtest_results['trade_win_rate_pct'].mean()
            avg_sharpe = backtest_results['sharpe_per_bar'].mean()

            # FPR analysis from experiment log (not confusion metrics)
            avg_fpr = experiment_log['fpr'].mean()
            min_fpr = experiment_log['fpr'].min()

            profitable_models = (backtest_results['total_return_net_pct'] > 0).sum()
            total_models = len(backtest_results)

            result = {
                'training_period_months': period_months,
                'data_rows': len(period_data),
                'start_date': period_data['datetime'].min(),
                'end_date': period_data['datetime'].max(),
                'avg_return_net_pct': avg_return,
                'std_return_net_pct': std_return,
                'max_return_net_pct': max_return,
                'min_return_net_pct': min_return,
                'avg_trades_count': avg_trades,
                'avg_win_rate_pct': avg_win_rate,
                'avg_sharpe_per_bar': avg_sharpe,
                'avg_fpr': avg_fpr,
                'min_fpr': min_fpr,
                'profitable_models': profitable_models,
                'total_models': total_models,
                'profitable_ratio': profitable_models / total_models
            }

            results.append(result)

            print(f"Results for {period_months}m:")
            print(f"  Avg Return: {avg_return:.2f}% (std: {std_return:.2f}%)")
            print(f"  Avg Trades: {avg_trades:.1f}")
            print(f"  Avg Win Rate: {avg_win_rate:.1f}%")
            print(f"  Avg FPR: {avg_fpr:.3f}")
            print(f"  Profitable Models: {profitable_models}/{total_models} ({profitable_models/total_models:.1%})")

        except Exception as e:
            print(f"Error with {period_months}-month period: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('catalysis/training_period_analysis.csv', index=False)

    print(f"\n=== SUMMARY ===")
    print(f"Results saved to catalysis/training_period_analysis.csv")
    print("\nTraining Period Analysis:")
    print(results_df[['training_period_months', 'avg_return_net_pct', 'avg_fpr', 'profitable_ratio']].to_string(index=False))

    # Analysis recommendations
    print(f"\n=== RECOMMENDATIONS ===")

    # Find periods with positive average returns
    positive_periods = results_df[results_df['avg_return_net_pct'] > 0]
    if len(positive_periods) > 0:
        min_viable = positive_periods['training_period_months'].min()
        max_viable = positive_periods['training_period_months'].max()
        print(f"Viable training period range: {min_viable}-{max_viable} months")

        # Find sweet spot (high returns, low FPR, good profitability ratio)
        sweet_spot = positive_periods[
            (positive_periods['avg_fpr'] < 0.1) &  # Low FPR
            (positive_periods['profitable_ratio'] > 0.3)  # Good success rate
        ]

        if len(sweet_spot) > 0:
            best_period = sweet_spot.loc[sweet_spot['avg_return_net_pct'].idxmax()]
            print(f"Recommended sweet spot: {best_period['training_period_months']} months")
            print(f"  Return: {best_period['avg_return_net_pct']:.2f}%")
            print(f"  FPR: {best_period['avg_fpr']:.3f}")
            print(f"  Success Rate: {best_period['profitable_ratio']:.1%}")

    return results_df


if __name__ == "__main__":
    results = analyze_training_periods()