import uuid
import sys
import loop

from loop import sfm

from loop.tests.utils.cleanup import cleanup_csv_files
from loop.tests.utils.get_data import get_klines_data, get_trades_data


def test_sfm():
    
    tests = [
        (sfm.random, get_klines_data, True),
        (sfm.xgboost, get_klines_data, False),
        (sfm.logreg_reference, get_klines_data, True),
        (sfm.logreg.regime_multiclass, get_klines_data, True),
        (sfm.logreg.breakout_regressor_ridge, get_klines_data, True),
        (sfm.lightgbm_reference, get_trades_data, False),
        (sfm.lightgbm.regime_multiclass, get_klines_data, False),
        ]

    for test in tests:
        
        try:
            uel = loop.UniversalExperimentLoop(data=test[1](),
                                                single_file_model=test[0])
            uel.run(experiment_name=uuid.uuid4().hex[:8],
                    n_permutations=2,
                    prep_each_round=test[2])
            print(f'    ✅ {test[0].__name__}: PASSED')
        
        except Exception as e:
            print(f'    ❌ {test[0].__name__}: FAILED - {e}')
            cleanup_csv_files()
            sys.exit(1)