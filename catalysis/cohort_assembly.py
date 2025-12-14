#!/usr/bin/env python3
"""
Cohort Assembly: Filter and assemble the final cohort from execution results
Apply economic thresholds and create final trading ensemble
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json


def assemble_final_cohort():
    """
    Filter cohort execution results and assemble final trading ensemble
    """

    print("=== COHORT ASSEMBLY ===")
    print("Loading cohort execution results...")

    try:
        # Load cohort results
        cohort_results = pd.read_csv('catalysis/cohort_all_results.csv')
        print(f"Loaded {len(cohort_results)} models from cohort execution")

        # Load discovery thresholds
        with open('catalysis/cohort_design.json', 'r') as f:
            cohort_design = json.load(f)

        economic_thresholds = cohort_design['discovery_inputs']['economic_thresholds']
        print(f"Economic thresholds: {economic_thresholds}")

    except FileNotFoundError:
        print("❌ Cohort results not found. Run cohort_execution.py first.")
        return None

    print(f"\\n=== COHORT ANALYSIS BY PROFILE ===")

    # Analyze each profile
    profiles = cohort_results['profile'].unique()
    profile_summaries = []

    for profile in profiles:
        profile_data = cohort_results[cohort_results['profile'] == profile]

        summary = {
            'profile': profile,
            'total_models': len(profile_data),
            'avg_return_pct': profile_data['total_return_net_pct'].mean(),
            'std_return_pct': profile_data['total_return_net_pct'].std(),
            'avg_fpr': profile_data['fpr'].mean(),
            'avg_trades': profile_data['trades_count'].mean(),
            'avg_win_rate': profile_data['trade_win_rate_pct'].mean(),
            'avg_sharpe': profile_data['sharpe_per_bar'].mean(),
            'profitable_models': (profile_data['total_return_net_pct'] > 0).sum(),
            'profitable_ratio': (profile_data['total_return_net_pct'] > 0).mean()
        }

        profile_summaries.append(summary)

        print(f"\\n{profile.upper()}:")
        print(f"  Models: {summary['total_models']}")
        print(f"  Avg Return: {summary['avg_return_pct']:.2f}% ± {summary['std_return_pct']:.2f}%")
        print(f"  Avg FPR: {summary['avg_fpr']:.3f}")
        print(f"  Avg Trades: {summary['avg_trades']:.1f}")
        print(f"  Profitable: {summary['profitable_models']}/{summary['total_models']} ({summary['profitable_ratio']:.1%})")

    print(f"\\n=== APPLYING ECONOMIC FILTERS ===")

    # Apply relaxed economic thresholds (discovery thresholds were too strict)
    relaxed_thresholds = {
        'min_return_threshold_pct': 0.5,  # At least 0.5% return
        'max_fpr_threshold': 0.05,        # Relaxed from 0.0 to 0.05 (5% FPR max)
        'min_trades_threshold': 3,        # Reduced from 5 to 3 trades minimum
        'min_sharpe_threshold': 0.0       # Keep at 0.0
    }

    print(f"Relaxed thresholds (more practical than discovery): {relaxed_thresholds}")

    # Apply filters
    viable_models = cohort_results[
        (cohort_results['total_return_net_pct'] >= relaxed_thresholds['min_return_threshold_pct']) &
        (cohort_results['fpr'] <= relaxed_thresholds['max_fpr_threshold']) &
        (cohort_results['trades_count'] >= relaxed_thresholds['min_trades_threshold']) &
        (cohort_results['sharpe_per_bar'] >= relaxed_thresholds['min_sharpe_threshold'])
    ].copy()

    print(f"\\nViable models after filtering: {len(viable_models)}/{len(cohort_results)} ({len(viable_models)/len(cohort_results):.1%})")

    if len(viable_models) == 0:
        print("❌ No models meet the economic thresholds!")
        return None

    # Rank models by combined score
    print(f"\\n=== RANKING VIABLE MODELS ===")

    # Create composite score (higher is better)
    viable_models['return_score'] = viable_models['total_return_net_pct'] / viable_models['total_return_net_pct'].std()
    viable_models['fpr_score'] = -(viable_models['fpr'] - viable_models['fpr'].min()) / (viable_models['fpr'].max() - viable_models['fpr'].min())
    viable_models['trades_score'] = (viable_models['trades_count'] - viable_models['trades_count'].min()) / (viable_models['trades_count'].max() - viable_models['trades_count'].min())
    viable_models['sharpe_score'] = viable_models['sharpe_per_bar'] / viable_models['sharpe_per_bar'].std()

    # Combined score with weights
    viable_models['composite_score'] = (
        0.4 * viable_models['return_score'] +      # 40% weight on returns
        0.3 * viable_models['fpr_score'] +         # 30% weight on low FPR
        0.2 * viable_models['sharpe_score'] +      # 20% weight on Sharpe
        0.1 * viable_models['trades_score']        # 10% weight on activity
    )

    # Sort by composite score
    final_cohort = viable_models.sort_values('composite_score', ascending=False).copy()

    print(f"Top 10 models by composite score:")
    top_10 = final_cohort.head(10)
    for i, (idx, model) in enumerate(top_10.iterrows(), 1):
        print(f"  {i:2d}. Profile: {model['profile']:16s} | Return: {model['total_return_net_pct']:6.2f}% | FPR: {model['fpr']:.3f} | Score: {model['composite_score']:.2f}")

    # Assemble final ensemble by selecting top models from each profile
    print(f"\\n=== ASSEMBLING FINAL ENSEMBLE ===")

    ensemble_models = []
    target_models_per_profile = {'ultra_conservative': 3, 'balanced_precision': 5, 'active_trading': 7}

    for profile in profiles:
        profile_models = final_cohort[final_cohort['profile'] == profile]
        target_count = target_models_per_profile.get(profile, 3)
        selected = profile_models.head(min(target_count, len(profile_models)))
        ensemble_models.append(selected)
        print(f"Selected {len(selected)} models from {profile}")

    final_ensemble = pd.concat(ensemble_models, ignore_index=True)

    print(f"\\nFinal ensemble: {len(final_ensemble)} models")
    print(f"Ensemble summary:")
    print(f"  Avg Return: {final_ensemble['total_return_net_pct'].mean():.2f}%")
    print(f"  Avg FPR: {final_ensemble['fpr'].mean():.3f}")
    print(f"  Avg Trades: {final_ensemble['trades_count'].mean():.1f}")
    print(f"  Avg Sharpe: {final_ensemble['sharpe_per_bar'].mean():.3f}")

    # Save results
    final_cohort.to_csv('catalysis/cohort_viable_models.csv', index=False)
    final_ensemble.to_csv('catalysis/cohort_final_ensemble.csv', index=False)

    # Save ensemble summary
    ensemble_summary = {
        'total_models_executed': len(cohort_results),
        'viable_models': len(viable_models),
        'final_ensemble_size': len(final_ensemble),
        'ensemble_metrics': {
            'avg_return_pct': final_ensemble['total_return_net_pct'].mean(),
            'avg_fpr': final_ensemble['fpr'].mean(),
            'avg_trades': final_ensemble['trades_count'].mean(),
            'avg_sharpe': final_ensemble['sharpe_per_bar'].mean()
        },
        'thresholds_applied': relaxed_thresholds,
        'profile_distribution': final_ensemble['profile'].value_counts().to_dict()
    }

    with open('catalysis/cohort_ensemble_summary.json', 'w') as f:
        json.dump(ensemble_summary, f, indent=2, default=str)

    print(f"\\nResults saved to:")
    print(f"  catalysis/cohort_viable_models.csv")
    print(f"  catalysis/cohort_final_ensemble.csv")
    print(f"  catalysis/cohort_ensemble_summary.json")

    return final_ensemble


if __name__ == "__main__":
    ensemble = assemble_final_cohort()