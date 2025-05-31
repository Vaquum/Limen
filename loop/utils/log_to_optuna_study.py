import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import create_trial, TrialState
from optuna.storages import InMemoryStorage


def log_to_optuna_study(log_df, params, objective):

    '''Creates an Optuna study from the Loop experiment artifacts.

    Args:
        log_df (uel.log_df | pl.DataFrame) : The experiment result log
        params (sfm.params | func) : sfm.params function used in the experiment
        objective (str) : Target feature column name
    
    '''

    distributions = {name: CategoricalDistribution(values) for name, values in params.items()}
    
    storage = InMemoryStorage()
    new_study = optuna.create_study(storage=storage, direction="minimize")
    
    param_cols = list(params.keys())
    
    for _, row in log_df.to_pandas().iterrows():
        
        params_dict = {c: row[c] for c in param_cols}
        
        trial = create_trial(params = params_dict,
                             distributions = distributions,
                             value = row[objective],
                             state = TrialState.COMPLETE)
        
        new_study.add_trial(trial)

    return new_study
