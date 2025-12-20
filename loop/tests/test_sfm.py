import uuid
import sys
import traceback
import pandas as pd
import polars as pl

import loop
from loop import sfm
from loop.tests.utils.cleanup import cleanup_csv_files
from loop.tests.utils.get_data import get_klines_data_fast



def test_sfm():

    tests = [
        # COLUMN ORDER: sfm, data_endpoint, prep_each_round
        # Manifest-driven SFMs (data_endpoint=None, auto-fetch from manifest)
        (sfm.reference.random, None, True),
        (sfm.reference.xgboost, None, True),
        (sfm.reference.logreg, None, True),
        (sfm.logreg.regime_multiclass, None, True),
        (sfm.logreg.breakout_regressor_ridge, None, True),
        (sfm.reference.lightgbm, None, True),
        (sfm.lightgbm.tradeable_regressor, None, True),
        (sfm.rules_based.momentum_volatility_longonly, None, True),
        (sfm.rules_based.momentum_volatility, None, True),
        # Legacy SFMs (no manifest or require explicit data)
        (sfm.ridge.ridge_classifier, get_klines_data_fast, True),
        (sfm.lightgbm.tradeline_long_binary, get_klines_data_fast, True),
        (sfm.lightgbm.tradeline_multiclass, get_klines_data_fast, True),
        (sfm.lightgbm.tradeline_directional_conditional, get_klines_data_fast, True),
    ]

    for test in tests:

        try:

            if test[1] is not None:
                uel = loop.UniversalExperimentLoop(data=test[1](),
                                                   single_file_model=test[0])
            else:
                uel = loop.UniversalExperimentLoop(single_file_model=test[0])

            experiment_name = uuid.uuid4().hex[:8]

            uel.run(experiment_name=experiment_name,
                    n_permutations=2,
                    prep_each_round=test[2])

            print(f'    ✅ {test[0].__name__}: PASSED')

        except Exception as e:

            print(f'    ❌ {test[0].__name__}: FAILED - {e}')

            cleanup_csv_files()
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":

    test_sfm()
