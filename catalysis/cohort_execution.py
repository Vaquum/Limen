#!/usr/bin/env python3
"""
Cohort Execution: Run multiple specialized models based on cohort design
Execute 3 cohort profiles with 180 total models
"""

import warnings
warnings.filterwarnings('ignore')

import loop
import polars as pl
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta


def create_profile_sfm(profile_name, profile_config, base_config):
    """
    Create a custom SFM for a specific cohort profile
    """

    class ProfileSFM:
        def __init__(self, profile_name, profile_config, base_config):
            self.profile_name = profile_name
            self.profile_config = profile_config
            self.base_config = base_config

        def params(self):
            """Generate parameter combinations for this profile"""
            params = {}

            # Start with base config
            for param, values in self.base_config.items():
                if param == 'training_period_months':
                    continue  # Handle separately
                params[param] = values if isinstance(values, list) else [values]

            # Override with profile-specific constraints
            for param, values in self.profile_config.items():
                if param not in ['description', 'samples_per_confidence']:
                    params[param] = values

            return params

        def prep(self, data, round_params=None):
            return loop.sfm.lightgbm.tradeline_long_binary.prep(data, round_params)

        def model(self, data, round_params):
            return loop.sfm.lightgbm.tradeline_long_binary.model(data, round_params)

    return ProfileSFM(profile_name, profile_config, base_config)


def execute_cohort():
    """
    Execute the complete cohort strategy based on discovery results
    """

    print("=== COHORT EXECUTION ===")
    print("Loading cohort design...")

    # Load cohort design
    with open('catalysis/cohort_design.json', 'r') as f:
        cohort_design = json.load(f)

    base_config = cohort_design['base_config']
    cohort_profiles = cohort_design['cohort_profiles']

    print(f"Total cohort size: {cohort_design['total_cohort_size']} models")
    print(f"Profiles: {list(cohort_profiles.keys())}")

    # Load historical data with 30-month training period
    print("\\nLoading historical data with 30-month training period...")
    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=3600, start_date_limit='2023-03-01')  # 30 months from ~Aug 2025

    print(f"Dataset: {len(historical.data)} rows from {historical.data['datetime'].min()} to {historical.data['datetime'].max()}")

    # Execute each profile
    all_results = []
    all_experiment_logs = []

    for profile_name, profile_config in cohort_profiles.items():
        print(f"\\n=== EXECUTING {profile_name.upper()} PROFILE ===")
        print(f"Description: {profile_config['description']}")

        # Calculate number of permutations for this profile
        n_confidence_levels = len(profile_config['confidence_threshold'])
        samples_per_conf = profile_config['samples_per_confidence']
        total_permutations = n_confidence_levels * samples_per_conf

        print(f"Confidence levels: {profile_config['confidence_threshold']}")
        print(f"Samples per confidence: {samples_per_conf}")
        print(f"Total permutations: {total_permutations}")

        # Create profile-specific SFM
        profile_sfm = create_profile_sfm(profile_name, profile_config, base_config)

        try:
            # Run UEL for this profile
            uel = loop.UniversalExperimentLoop(
                data=historical.data,
                single_file_model=profile_sfm
            )

            uel.run(
                experiment_name=f"cohort_{profile_name}",
                n_permutations=total_permutations,
                prep_each_round=True,
                save_to_sqlite=False
            )

            # Extract results
            backtest_results = uel.experiment_backtest_results
            experiment_log = uel.experiment_log

            print(f"✅ Profile {profile_name} completed: {len(backtest_results)} models")

            # Add profile identifier to results
            backtest_results['profile'] = profile_name
            experiment_log = experiment_log.with_columns(pl.lit(profile_name).alias('profile'))

            # Store results
            all_results.append(backtest_results)
            all_experiment_logs.append(experiment_log)

            # Quick profile summary
            avg_return = backtest_results['total_return_net_pct'].mean()
            avg_fpr = experiment_log['fpr'].mean()
            profitable_count = (backtest_results['total_return_net_pct'] > 0).sum()

            print(f"Profile {profile_name} summary:")
            print(f"  Avg return: {avg_return:.2f}%")
            print(f"  Avg FPR: {avg_fpr:.3f}")
            print(f"  Profitable models: {profitable_count}/{len(backtest_results)}")

        except Exception as e:
            print(f"❌ Error in profile {profile_name}: {e}")
            continue

    if not all_results:
        print("❌ No profiles completed successfully!")
        return None, None

    # Combine all results
    print("\\n=== COMBINING COHORT RESULTS ===")
    cohort_backtest_results = pd.concat(all_results, ignore_index=True)
    cohort_experiment_log = pl.concat(all_experiment_logs)

    print(f"Total cohort models: {len(cohort_backtest_results)}")

    # Add FPR and other metrics to backtest results
    cohort_backtest_results['fpr'] = cohort_experiment_log['fpr'].to_pandas()
    cohort_backtest_results['precision'] = cohort_experiment_log['precision'].to_pandas()
    cohort_backtest_results['recall'] = cohort_experiment_log['recall'].to_pandas()
    cohort_backtest_results['f1_score'] = cohort_experiment_log['f1'].to_pandas()

    # Save complete cohort results
    cohort_backtest_results.to_csv('catalysis/cohort_all_results.csv', index=False)
    cohort_experiment_log.write_csv('catalysis/cohort_experiment_log.csv')

    print("\\nCohort execution completed!")
    print("Results saved to:")
    print("  catalysis/cohort_all_results.csv")
    print("  catalysis/cohort_experiment_log.csv")

    return cohort_backtest_results, cohort_experiment_log


if __name__ == "__main__":
    backtest_results, experiment_log = execute_cohort()