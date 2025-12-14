#!/usr/bin/env python3
"""
Debug Permutation 38

Simulates the UEL random parameter selection to see what parameters
are chosen for permutation 38.
"""

import sys
sys.path.insert(0, '/Users/beyondsyntax/Loop/catalysis')
sys.path.insert(0, '/Users/beyondsyntax/Loop')

import numpy as np
import directional_conditional as dc

# Get the parameter grid
params = dc.params()

# Simulate UEL random search with random_state=42 (from CONFIG)
np.random.seed(42)

# Get all parameter names and their values
param_names = list(params.keys())
param_values = [params[k] for k in param_names]

print('=' * 80)
print('PARAMETER GRID ANALYSIS')
print('=' * 80)

print('\nParameter Space:')
for name in param_names:
    values = params[name]
    print(f'  {name:25s}: {values} ({len(values)} options)')

total_combos = np.prod([len(v) for v in param_values])
print(f'\nTotal possible combinations: {total_combos:,}')

print('\n' + '=' * 80)
print('SIMULATING 50 RANDOM PERMUTATIONS')
print('=' * 80)

n_permutations = 50

# Generate permutations the same way UEL does
permutations = []
for i in range(n_permutations):
    # Random permutation: pick one value from each parameter
    perm = {}
    for name in param_names:
        values = params[name]
        perm[name] = np.random.choice(values)
    permutations.append(perm)

    if i == 37:  # Permutation 38 (0-indexed)
        print(f'\n*** PERMUTATION 38 (index {i}) ***')
        print('Parameters:')
        for name, value in perm.items():
            print(f'  {name:25s}: {value}')

print('\n' + '=' * 80)
print('ANALYZING ALL PERMUTATIONS')
print('=' * 80)

# Look for patterns
print('\nPermutation parameter values:')
for i in range(min(50, len(permutations))):
    perm = permutations[i]
    print(f'\n  Permutation {i+1}:')
    print(f'    threshold_pct={perm["threshold_pct"]}, lookahead_hours={perm["lookahead_hours"]}, use_safer={perm["use_safer"]}')
    print(f'    conditional_threshold={perm["conditional_threshold"]}, movement_threshold={perm["movement_threshold"]}')

print('\n' + '=' * 80)
print('CHECKING FOR PROBLEMATIC COMBINATIONS')
print('=' * 80)

perm_38 = permutations[37]

print('\nPermutation 38 Analysis:')
print(f'  threshold_pct: {perm_38["threshold_pct"]}')
print(f'  lookahead_hours: {perm_38["lookahead_hours"]}')
print(f'  quantile_threshold: {perm_38["quantile_threshold"]}')
print(f'  min_height_pct: {perm_38["min_height_pct"]}')
print(f'  max_duration_hours: {perm_38["max_duration_hours"]}')
print(f'  conditional_threshold: {perm_38["conditional_threshold"]}')
print(f'  movement_threshold: {perm_38["movement_threshold"]}')
print(f'  use_safer: {perm_38["use_safer"]}')
print(f'  n_estimators: {perm_38["n_estimators"]}')
print(f'  num_leaves: {perm_38["num_leaves"]}')
print(f'  learning_rate: {perm_38["learning_rate"]}')

# Check for potential issues
issues = []

if perm_38['threshold_pct'] < perm_38['min_height_pct']:
    issues.append(f"threshold_pct ({perm_38['threshold_pct']}) < min_height_pct ({perm_38['min_height_pct']})")

if perm_38['max_duration_hours'] < perm_38['lookahead_hours']:
    issues.append(f"max_duration_hours ({perm_38['max_duration_hours']}) < lookahead_hours ({perm_38['lookahead_hours']})")

if issues:
    print('\n⚠️  POTENTIAL ISSUES DETECTED:')
    for issue in issues:
        print(f'    - {issue}')
else:
    print('\n✓ No obvious parameter conflicts detected')

print('\n' + '=' * 80)
print('Done!')
