#!/usr/bin/env python3
"""
Discovery Run 2: Economic Viability Thresholds Analysis
Analyze UEL's built-in economic metrics to establish viability thresholds for cohort filtering
"""

import warnings
warnings.filterwarnings('ignore')

import loop
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def analyze_economic_thresholds():
    """
    Run tradeline_long_binary with broader parameter space to understand:
    1. Distribution of economic metrics from UEL backtest results
    2. Natural thresholds for economic viability
    3. Transaction cost impact patterns
    4. Pattern density vs profitability trade-offs
    """

    print("Loading historical data...")
    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=3600, start_date_limit='2022-01-01')  # Use recent 2+ years

    print(f"Dataset: {len(historical.data)} rows from {historical.data['datetime'].min()} to {historical.data['datetime'].max()}")

    print("\n=== Running Economic Threshold Discovery ===")
    print("Using larger parameter space to understand economic distributions...")

    try:
        # Run UEL with larger parameter space for economic analysis
        uel = loop.UniversalExperimentLoop(
            data=historical.data,
            single_file_model=loop.sfm.lightgbm.tradeline_long_binary
        )

        # Use more permutations to get good statistical distribution
        uel.run(
            experiment_name="discovery_economic_thresholds",
            n_permutations=500,  # Larger sample for economic analysis
            prep_each_round=True,
            save_to_sqlite=False
        )

        # Extract economic metrics
        backtest_results = uel.experiment_backtest_results
        experiment_log = uel.experiment_log

        print(f"\nAnalyzing {len(backtest_results)} model permutations...")

        # Economic Analysis
        print("\n=== ECONOMIC METRICS DISTRIBUTION ===")

        # Returns analysis
        returns = backtest_results['total_return_net_pct']
        print(f"Net Returns (%):")
        print(f"  Mean: {returns.mean():.2f}%")
        print(f"  Median: {returns.median():.2f}%")
        print(f"  Std: {returns.std():.2f}%")
        print(f"  Min: {returns.min():.2f}%")
        print(f"  Max: {returns.max():.2f}%")
        print(f"  25th percentile: {returns.quantile(0.25):.2f}%")
        print(f"  75th percentile: {returns.quantile(0.75):.2f}%")

        # Profitability analysis
        profitable_models = (returns > 0).sum()
        breakeven_models = (returns == 0).sum()
        losing_models = (returns < 0).sum()
        total_models = len(returns)

        print(f"\nProfitability Distribution:")
        print(f"  Profitable: {profitable_models}/{total_models} ({profitable_models/total_models:.1%})")
        print(f"  Breakeven: {breakeven_models}/{total_models} ({breakeven_models/total_models:.1%})")
        print(f"  Losing: {losing_models}/{total_models} ({losing_models/total_models:.1%})")

        # Trading activity analysis
        trades = backtest_results['trades_count']
        print(f"\nTrading Activity:")
        print(f"  Mean trades: {trades.mean():.1f}")
        print(f"  Median trades: {trades.median():.1f}")
        print(f"  Min trades: {trades.min()}")
        print(f"  Max trades: {trades.max()}")

        # Win rate analysis
        win_rates = backtest_results['trade_win_rate_pct']
        print(f"\nWin Rates (%):")
        print(f"  Mean: {win_rates.mean():.1f}%")
        print(f"  Median: {win_rates.median():.1f}%")
        print(f"  75th percentile: {win_rates.quantile(0.75):.1f}%")

        # Sharpe analysis
        sharpe_values = backtest_results['sharpe_per_bar']
        print(f"\nSharpe per Bar:")
        print(f"  Mean: {sharpe_values.mean():.3f}")
        print(f"  Median: {sharpe_values.median():.3f}")
        print(f"  75th percentile: {sharpe_values.quantile(0.75):.3f}")

        # Transaction cost analysis
        if 'cost_round_trip_bps' in backtest_results.columns:
            costs = backtest_results['cost_round_trip_bps']
            print(f"\nTransaction Costs (bps):")
            print(f"  Mean: {costs.mean():.1f}")
            print(f"  Median: {costs.median():.1f}")

        # FPR analysis from experiment log (not confusion metrics)
        fpr_values = experiment_log['fpr']
        print(f"\nFalse Positive Rates:")
        print(f"  Mean: {fpr_values.mean():.3f}")
        print(f"  Median: {fpr_values.median():.3f}")
        print(f"  25th percentile: {fpr_values.quantile(0.25):.3f}")
        print(f"  10th percentile: {fpr_values.quantile(0.10):.3f}")

        # Cross-analysis: Economic performance by FPR
        print("\n=== ECONOMIC PERFORMANCE BY FPR QUARTILES ===")

        # Merge metrics for analysis
        analysis_df = backtest_results.copy()
        analysis_df['fpr'] = experiment_log['fpr']

        # FPR quartiles
        fpr_q1 = fpr_values.quantile(0.25)
        fpr_q2 = fpr_values.quantile(0.50)
        fpr_q3 = fpr_values.quantile(0.75)

        low_fpr = analysis_df[analysis_df['fpr'] <= fpr_q1]
        med_fpr = analysis_df[(analysis_df['fpr'] > fpr_q1) & (analysis_df['fpr'] <= fpr_q3)]
        high_fpr = analysis_df[analysis_df['fpr'] > fpr_q3]

        print(f"Low FPR (≤{fpr_q1:.3f}): {len(low_fpr)} models")
        print(f"  Avg Return: {low_fpr['total_return_net_pct'].mean():.2f}%")
        print(f"  Profitable: {(low_fpr['total_return_net_pct'] > 0).sum()}/{len(low_fpr)} ({(low_fpr['total_return_net_pct'] > 0).mean():.1%})")
        print(f"  Avg Trades: {low_fpr['trades_count'].mean():.1f}")

        print(f"\nMedium FPR ({fpr_q1:.3f}-{fpr_q3:.3f}): {len(med_fpr)} models")
        print(f"  Avg Return: {med_fpr['total_return_net_pct'].mean():.2f}%")
        print(f"  Profitable: {(med_fpr['total_return_net_pct'] > 0).sum()}/{len(med_fpr)} ({(med_fpr['total_return_net_pct'] > 0).mean():.1%})")
        print(f"  Avg Trades: {med_fpr['trades_count'].mean():.1f}")

        print(f"\nHigh FPR (>{fpr_q3:.3f}): {len(high_fpr)} models")
        print(f"  Avg Return: {high_fpr['total_return_net_pct'].mean():.2f}%")
        print(f"  Profitable: {(high_fpr['total_return_net_pct'] > 0).sum()}/{len(high_fpr)} ({(high_fpr['total_return_net_pct'] > 0).mean():.1%})")
        print(f"  Avg Trades: {high_fpr['trades_count'].mean():.1f}")

        # Save detailed results
        analysis_df.to_csv('catalysis/economic_threshold_analysis.csv', index=False)

        print(f"\n=== COHORT FILTERING RECOMMENDATIONS ===")

        # Recommended thresholds based on analysis
        min_return_threshold = max(0.5, returns.quantile(0.6))  # At least 60th percentile or 0.5%
        max_fpr_threshold = fpr_values.quantile(0.25)  # Bottom quartile FPR
        min_trades_threshold = max(5, trades.quantile(0.25))  # At least 25th percentile or 5 trades
        min_sharpe_threshold = sharpe_values.quantile(0.5)  # Median Sharpe

        print(f"Recommended minimum return threshold: {min_return_threshold:.2f}%")
        print(f"Recommended maximum FPR threshold: {max_fpr_threshold:.3f}")
        print(f"Recommended minimum trades threshold: {min_trades_threshold:.0f}")
        print(f"Recommended minimum Sharpe threshold: {min_sharpe_threshold:.3f}")

        # Test these thresholds
        viable_models = analysis_df[
            (analysis_df['total_return_net_pct'] >= min_return_threshold) &
            (analysis_df['fpr'] <= max_fpr_threshold) &
            (analysis_df['trades_count'] >= min_trades_threshold) &
            (analysis_df['sharpe_per_bar'] >= min_sharpe_threshold)
        ]

        print(f"\nUsing recommended thresholds:")
        print(f"  Viable models: {len(viable_models)}/{len(analysis_df)} ({len(viable_models)/len(analysis_df):.1%})")

        if len(viable_models) > 0:
            print(f"  Avg return of viable models: {viable_models['total_return_net_pct'].mean():.2f}%")
            print(f"  Avg FPR of viable models: {viable_models['fpr'].mean():.3f}")
            print(f"  Avg trades of viable models: {viable_models['trades_count'].mean():.1f}")

        # Save thresholds
        thresholds = {
            'min_return_threshold_pct': min_return_threshold,
            'max_fpr_threshold': max_fpr_threshold,
            'min_trades_threshold': min_trades_threshold,
            'min_sharpe_threshold': min_sharpe_threshold,
            'viable_model_count': len(viable_models),
            'total_model_count': len(analysis_df),
            'viable_ratio': len(viable_models) / len(analysis_df)
        }

        thresholds_df = pd.DataFrame([thresholds])
        thresholds_df.to_csv('catalysis/economic_thresholds.csv', index=False)

        print(f"\nResults saved to:")
        print(f"  catalysis/economic_threshold_analysis.csv")
        print(f"  catalysis/economic_thresholds.csv")

        return analysis_df, thresholds

    except Exception as e:
        print(f"Error in economic threshold analysis: {e}")
        raise


if __name__ == "__main__":
    analysis_df, thresholds = analyze_economic_thresholds()