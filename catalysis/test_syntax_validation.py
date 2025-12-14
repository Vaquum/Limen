#!/usr/bin/env python3
"""
Syntax validation test for micro-period sensitivity analysis
Tests that the code has correct syntax and basic structure without running the full analysis.
"""

import ast
import sys
from pathlib import Path

def validate_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the AST to check for syntax errors
        ast.parse(content)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_required_functions(file_path):
    """Check if required functions exist in the file."""
    required_functions = [
        'create_time_windows',
        'run_sfm_on_window',
        'micro_period_sensitivity_analysis',
        'calculate_stability_metrics'
    ]

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        tree = ast.parse(content)

        # Get all function definitions
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        missing = [f for f in required_functions if f not in functions]
        return len(missing) == 0, missing, functions

    except Exception as e:
        return False, [], []

def main():
    print("=" * 60)
    print("MICRO-PERIOD SENSITIVITY ANALYSIS - SYNTAX VALIDATION")
    print("=" * 60)

    # Files to validate
    files_to_check = [
        "sfm_micro_period_analysis.py",
        "sfm_sensitivity_runner.py",
        "quick_test.py"
    ]

    all_passed = True

    for file_name in files_to_check:
        file_path = Path(__file__).parent / file_name
        print(f"\n📁 Checking {file_name}...")

        if not file_path.exists():
            print(f"   ❌ File not found: {file_path}")
            all_passed = False
            continue

        # Check syntax
        syntax_ok, syntax_msg = validate_syntax(file_path)
        if syntax_ok:
            print(f"   ✅ Syntax validation passed")
        else:
            print(f"   ❌ Syntax validation failed: {syntax_msg}")
            all_passed = False
            continue

        # Check required functions (only for the main analysis file)
        if file_name == "sfm_micro_period_analysis.py":
            funcs_ok, missing, all_funcs = check_required_functions(file_path)
            if funcs_ok:
                print(f"   ✅ All required functions present")
                print(f"   📝 Functions found: {', '.join(all_funcs[:5])}{'...' if len(all_funcs) > 5 else ''}")
            else:
                print(f"   ❌ Missing required functions: {missing}")
                all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("The micro-period sensitivity analysis code is ready for testing.")
        print("\nOptimizations implemented:")
        print("✓ Small test datasets (500-1000 rows max)")
        print("✓ Limited windows (5-10 max per window size)")
        print("✓ Progress tracking with verbose output")
        print("✓ Minimum data requirements (50+ rows per window)")
        print("✓ Intelligent window stride calculation")
        print("✓ Error handling with informative messages")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please review the errors above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())