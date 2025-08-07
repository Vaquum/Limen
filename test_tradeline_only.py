#!/usr/bin/env python3
"""
Quick test for tradeline_multiclass SFM only
"""

import sys
sys.path.append('/Users/beyondsyntax/Loop')

import loop
from loop import sfm
from loop.tests.utils.get_data import get_klines_data
import uuid

# Test tradeline_multiclass
print("Testing tradeline_multiclass SFM...")
data = get_klines_data()
print(f"Data shape: {data.shape}")

uel = loop.UniversalExperimentLoop(
    data=data,
    single_file_model=sfm.lightgbm.tradeline_multiclass
)

# Run with minimal permutations
print("Running experiment...")
uel.run(
    experiment_name='test_tradeline_' + uuid.uuid4().hex[:8],
    n_permutations=1,
    prep_each_round=True
)

print("\nResults:")
print(uel.log_df)

if uel.extras:
    extras = uel.extras[0]
    print(f"\nQuantile threshold: {extras['quantile_threshold']}")
    print(f"Lines found: {extras['n_long_lines']} long, {extras['n_short_lines']} short")
    print(f"Class distribution: {extras['class_distribution']['test']}")

print("\n✅ tradeline_multiclass test completed successfully!")