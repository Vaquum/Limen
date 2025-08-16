import uuid
import sys
import loop
import traceback

from loop import sfm
from loop.tests.utils.cleanup import cleanup_csv_files
from loop.tests.utils.get_data import get_klines_data, get_trades_data, get_klines_data_small


def test_sfm():
    
    tests = [
        # COLUMN ORDER: sfm, data_endpoint, prep_each_round, log
        (sfm.reference.random, get_klines_data, True, False),
        (sfm.reference.xgboost, get_klines_data, False, True),
        (sfm.reference.logreg, get_klines_data, True, True),
        (sfm.logreg.regime_multiclass, get_klines_data, False, True),
        (sfm.logreg.breakout_regressor_ridge, get_klines_data, False, False),
        (sfm.reference.lightgbm, get_klines_data, False, True),
        (sfm.lightgbm.tradeable_regressor, get_klines_data, False, False),
        (sfm.lightgbm.tradeline_multiclass, get_klines_data_small, True, False),
        (sfm.rules_based.momentum_volatility_longonly, get_klines_data_small, True, False),
        (sfm.rules_based.momentum_volatility, get_klines_data_small, True, False),
        ]

    for test in tests:
        
        try:
            
            uel = loop.UniversalExperimentLoop(data=test[1](),
                                                single_file_model=test[0])
            
            uel.run(experiment_name=uuid.uuid4().hex[:8],
                    n_permutations=2,
                    prep_each_round=test[2])

            if test[3]:
                _ = uel.log.experiment_backtest_results()
                _ = uel.log.experiment_confusion_metrics(x='price_change')
                _ = uel.log.permutation_prediction_performance(round_id=0)
                _ = uel.log.permutation_confusion_metrics(x='price_change', round_id=0)

            print(f'    ✅ {test[0].__name__}: PASSED')
        
        except Exception as e:
            
            print(f'    ❌ {test[0].__name__}: FAILED - {e}')
            
            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    
    test_sfm()
