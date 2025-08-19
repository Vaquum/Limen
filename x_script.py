import loop

import warnings
warnings.filterwarnings('ignore')

def get_uel():

    historical = loop.HistoricalData()
    historical.get_spot_klines(kline_size=3600, start_date_limit='2020-01-01')

    uel = loop.UniversalExperimentLoop(data=historical.data,
                                    single_file_model=loop.sfm.reference.logreg)
    
    uel.run(experiment_name=f"LogReg-Db0",
            n_permutations=50, 
            prep_each_round=True)
    
    return uel

#uel = get_uel()

#loop.explorer.loop_explorer(data=None, host='0.0.0.0', uel=uel)
