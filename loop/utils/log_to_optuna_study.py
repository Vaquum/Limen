import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import create_trial, TrialState
from optuna.storages import InMemoryStorage


def log_to_optuna_study(experiment_log: object, params: object, objective: str) -> optuna.Study:

    '''
    Create an Optuna study from the Loop experiment artifacts.

    Args:
        experiment_log (uel.experiment_log | pl.DataFrame) : The experiment result log
        params (sfd.params | func) : sfd.params function used in the experiment
        objective (str) : Target feature column name
        
    Returns:
        optuna.Study: The Optuna study object
    '''

    distributions = {name: CategoricalDistribution(values) for name, values in params.items()}
    
    storage = InMemoryStorage()
    new_study = optuna.create_study(storage=storage, direction="minimize")
    
    param_cols = list(params.keys())
    
    for row in experiment_log.iter_rows(named=True):

        params_dict = {c: row[c] for c in param_cols}

        trial = create_trial(params = params_dict,
                             distributions = distributions,
                             value = row[objective],
                             state = TrialState.COMPLETE)
        
        new_study.add_trial(trial)

    return new_study
