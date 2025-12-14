#!/usr/bin/env python3
"""
Debug FPR extraction from UEL confusion metrics
"""

import warnings
warnings.filterwarnings('ignore')

import loop
from loop.tests.utils.get_data import get_klines_data


def debug_fpr_extraction():
    """
    Debug why confusion metrics extraction fails in discovery script
    """

    print("Loading data...")
    data = get_klines_data()
    window_data = data.head(1000)

    print("\nRunning UEL...")
    uel = loop.UniversalExperimentLoop(
        data=window_data,
        single_file_model=loop.sfm.lightgbm.tradeline_long_binary
    )

    uel.run(
        experiment_name='debug_fpr_extraction',
        n_permutations=1,
        prep_each_round=True
    )

    print(f"UEL completed. Log shape: {uel.experiment_log.shape}")

    # Check experiment_log directly
    print("\n=== Direct experiment_log access ===")
    if len(uel.experiment_log) > 0:
        result_row = uel.experiment_log.row(0, named=True)
        print(f"FPR from experiment_log: {result_row.get('fpr', 'NOT_FOUND')}")
        print(f"Precision from experiment_log: {result_row.get('precision', 'NOT_FOUND')}")
        print(f"Recall from experiment_log: {result_row.get('recall', 'NOT_FOUND')}")

    # Check backtest results
    print("\n=== Backtest results access ===")
    try:
        backtest_results = uel.experiment_backtest_results
        print(f"Backtest results shape: {backtest_results.shape}")
        print(f"Backtest columns: {list(backtest_results.columns)}")
        if len(backtest_results) > 0:
            print(f"First row sample:\n{backtest_results.iloc[0]}")
    except Exception as e:
        print(f"Backtest results failed: {e}")

    # Check confusion metrics - this is where our discovery script fails
    print("\n=== Confusion metrics access ===")
    try:
        confusion_metrics = uel.experiment_confusion_metrics
        print(f"Confusion metrics shape: {confusion_metrics.shape}")
        print(f"Confusion columns: {list(confusion_metrics.columns)}")
        if len(confusion_metrics) > 0:
            print(f"FPR from confusion_metrics: {confusion_metrics['fpr'].iloc[0] if 'fpr' in confusion_metrics.columns else 'FPR_NOT_FOUND'}")
            print(f"First row sample:\n{confusion_metrics.iloc[0]}")
    except Exception as e:
        print(f"❌ Confusion metrics failed: {e}")
        import traceback
        traceback.print_exc()

    # Check manual confusion metrics call (what our discovery script does)
    print("\n=== Manual confusion metrics call ===")
    try:
        manual_confusion = uel._log.experiment_confusion_metrics('price_change')
        print(f"Manual confusion shape: {manual_confusion.shape}")
        print(f"Manual confusion columns: {list(manual_confusion.columns)}")
        if len(manual_confusion) > 0:
            fpr_value = manual_confusion['fpr'].iloc[0] if 'fpr' in manual_confusion.columns else 'FPR_NOT_FOUND'
            print(f"FPR from manual call: {fpr_value}")
    except Exception as e:
        print(f"❌ Manual confusion metrics failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_fpr_extraction()