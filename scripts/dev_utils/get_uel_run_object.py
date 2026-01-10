import limen

import warnings
warnings.filterwarnings('ignore')

def get_uel():

    historical = limen.HistoricalData()
    historical.get_spot_klines(kline_size=3600, start_date_limit='2020-01-01')

    uel = limen.UniversalExperimentLoop(data=historical.data,
                                    sfd=limen.sfd.foundational_sfd.logreg_binary)
    
    uel.run(experiment_name='LogReg-Db0',
            n_permutations=1000, 
            prep_each_round=True)
    
    return uel
