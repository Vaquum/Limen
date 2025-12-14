#!/usr/bin/env python3
"""
Cohort Design: Create parameter space for multiple specialized models
Based on discovery runs 1-2 results
"""

import warnings
warnings.filterwarnings('ignore')

import loop
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def design_cohort_parameter_space():
    """
    Design cohort parameter space based on discovery results:
    - Discovery Run 1: 30-month training period optimal
    - Discovery Run 2: Very strict thresholds (0.5% return, 0.0 FPR, 5 trades, 0.0 Sharpe)

    Strategy: Create multiple specialized models with different focuses:
    1. Ultra-conservative models (very low FPR)
    2. Balanced models (moderate FPR, good returns)
    3. Active models (higher trade frequency)
    """

    print("=== COHORT DESIGN BASED ON DISCOVERY RESULTS ===")
    print("Discovery Run 1: Optimal training period = 30 months")
    print("Discovery Run 2: Economic thresholds very strict (0.0 FPR requirement)")
    print("\nDesigning cohort with broader FPR tolerance for practical implementation...")

    # Base configuration from 30-month training period success
    base_config = {
        # Fixed based on discovery results
        'training_period_months': 30,  # From Discovery Run 1

        # Feature engineering - vary slightly for specialization
        'quantile_threshold': [0.70, 0.75, 0.80],  # Conservative to very conservative
        'min_height_pct': [0.002, 0.003, 0.004],   # Signal sensitivity
        'max_duration_hours': [36, 48, 60],        # Pattern duration
        'lookahead_hours': [36, 48, 60],           # Prediction horizon
        'long_threshold_percentile': [70, 75, 80], # Entry threshold

        # Trading parameters - create specialized profiles
        'position_size': [0.15, 0.20, 0.25],       # Risk sizing
        'min_stop_loss': [0.008, 0.010, 0.012],    # Conservative stops
        'max_stop_loss': [0.030, 0.040, 0.050],    # Maximum risk
        'atr_stop_multiplier': [1.0, 1.5, 2.0],    # ATR-based stops
        'trailing_activation': [0.015, 0.020, 0.025], # Profit protection
        'trailing_distance': [0.4, 0.5, 0.6],      # Trail distance
        'loser_timeout_hours': [18, 24, 30],       # Re-entry timeout
        'max_hold_hours': [36, 48, 60],            # Maximum position hold
        'default_atr_pct': [0.012, 0.015, 0.018],  # Default volatility

        # Model hyperparameters - create diverse ensemble
        'num_leaves': [31, 63, 127],               # Tree complexity
        'learning_rate': [0.05, 0.1, 0.15],       # Learning speed
        'feature_fraction': [0.8, 0.9, 1.0],      # Feature sampling
        'bagging_fraction': [0.8, 0.9, 1.0],      # Data sampling
        'bagging_freq': [3, 5, 7],                # Bagging frequency
        'min_child_samples': [15, 20, 25],        # Minimum leaf samples
        'lambda_l1': [0, 0.1, 0.2],               # L1 regularization
        'lambda_l2': [0, 0.1, 0.2],               # L2 regularization
        'n_estimators': [300, 500, 700],          # Number of trees

        # Confidence thresholds - key for FPR control
        'confidence_threshold': [
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
        ]
    }

    print(f"\nCohort parameter space designed:")
    total_combinations = 1
    for param, values in base_config.items():
        if isinstance(values, list):
            print(f"  {param}: {len(values)} options")
            total_combinations *= len(values)

    print(f"\nTotal possible combinations: {total_combinations:,}")
    print("Strategy: Generate cohort through targeted sampling")

    # Define cohort profiles
    cohort_profiles = {
        'ultra_conservative': {
            'description': 'Minimal FPR, very high confidence',
            'confidence_threshold': [0.75, 0.80, 0.85, 0.90],
            'quantile_threshold': [0.80],
            'position_size': [0.15],
            'min_stop_loss': [0.008],
            'lambda_l1': [0.1, 0.2],
            'lambda_l2': [0.1, 0.2],
            'samples_per_confidence': 10
        },

        'balanced_precision': {
            'description': 'Balance of precision and activity',
            'confidence_threshold': [0.55, 0.60, 0.65, 0.70],
            'quantile_threshold': [0.70, 0.75],
            'position_size': [0.20],
            'min_stop_loss': [0.010],
            'lambda_l1': [0, 0.1],
            'lambda_l2': [0, 0.1],
            'samples_per_confidence': 15
        },

        'active_trading': {
            'description': 'Higher activity, moderate confidence',
            'confidence_threshold': [0.45, 0.50, 0.55, 0.60],
            'quantile_threshold': [0.70],
            'position_size': [0.25],
            'min_stop_loss': [0.012],
            'lambda_l1': [0],
            'lambda_l2': [0],
            'samples_per_confidence': 20
        }
    }

    print(f"\n=== COHORT PROFILES ===")
    for profile_name, profile in cohort_profiles.items():
        print(f"\n{profile_name.upper()}:")
        print(f"  {profile['description']}")
        print(f"  Confidence range: {min(profile['confidence_threshold']):.2f} - {max(profile['confidence_threshold']):.2f}")
        print(f"  Samples per confidence: {profile['samples_per_confidence']}")
        total_samples = len(profile['confidence_threshold']) * profile['samples_per_confidence']
        print(f"  Total samples: {total_samples}")

    # Calculate total cohort size
    total_cohort_size = sum(
        len(profile['confidence_threshold']) * profile['samples_per_confidence']
        for profile in cohort_profiles.values()
    )

    print(f"\nTotal cohort size: {total_cohort_size} models")
    print(f"Estimated runtime: {total_cohort_size * 10 / 60:.1f} minutes")

    # Save cohort design
    cohort_design = {
        'base_config': base_config,
        'cohort_profiles': cohort_profiles,
        'total_cohort_size': total_cohort_size,
        'discovery_inputs': {
            'optimal_training_period': 30,
            'economic_thresholds': {
                'min_return_threshold_pct': 0.5,
                'max_fpr_threshold': 0.0,
                'min_trades_threshold': 5,
                'min_sharpe_threshold': 0.0
            }
        }
    }

    # Save as JSON for implementation
    import json
    with open('catalysis/cohort_design.json', 'w') as f:
        json.dump(cohort_design, f, indent=2, default=str)

    print(f"\nCohort design saved to catalysis/cohort_design.json")

    return cohort_design


if __name__ == "__main__":
    design = design_cohort_parameter_space()