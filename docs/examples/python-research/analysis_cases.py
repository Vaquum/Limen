"""Recorded candidate populations for analysis and metric-driven feedback."""
import json
from types import SimpleNamespace

import numpy as np
from limen import Log, UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.experiment.reducer import CorrelationReducer, FocusReducer, SanityReducer, SaturationReducer

import research as project


def recordable_prep(raw, round_params):
    data = project.prep(raw, round_params)
    validation = data['validation']
    data['_alignment'] = {'missing_datetimes': [], 'first_test_datetime': validation['datetime'][0], 'last_test_datetime': validation['datetime'][-1]}
    return data


# region analysis_results
def analyze_log(loop, directory, space):
    analysis = Log(uel_object=loop)
    correlation = analysis.experiment_parameter_correlation('validation_rmse', cols_to_drop=[column for column in project.completed_log(loop).columns if column not in {*space, 'validation_rmse'}],
        heads=(1.,), n_boot=50, min_n=10, random_state=42)
    importance = analysis.experiment_parameter_importance('validation_rmse', parameters=['window', 'alpha'], n_repeats=2, random_state=42)
    correlation.write_csv(directory / 'correlation.csv')
    importance.write_csv(directory / 'importance.csv')
    return correlation, importance
# endregion analysis_results


# region reducer_config
def metric_reducers():
    return {
        'sanity': SanityReducer(metric='validation_rmse', min_observations=2, execution_time_threshold=0., timeout_rate_threshold=0.),
        'saturation': SaturationReducer(metric='validation_rmse', cv_threshold=.1, min_samples_per_value=2, retain_fraction=.5),
        'focus': FocusReducer(metric='validation_rmse', maximize=False, breakthrough_threshold=1., min_observations=5, focus_timeout=2, variation_count=3),
        'correlation': CorrelationReducer(metric='validation_rmse', maximize=False, min_observations=10, n_boot=30,
            prune_threshold=1., sign_stability_threshold=0., random_state=42),
    }
# endregion reducer_config


# region analysis_campaign
def analyze_candidates(bars, output):
    space = {'window': list(range(4, 14)), 'alpha': [.01, .1, 1.]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        forecasts = fitted.predict(data['validation'].select('momentum'))
        return {'validation_rmse': float(np.sqrt(np.mean((forecasts - data['validation']['next_return'].to_numpy())**2))),
                'validation_net_return': project.backtest(data['validation'], forecasts)}

    def campaign(name, reducers):
        directory = output / name
        domain = ParamDomain(space)
        loop = UniversalExperimentLoop(data=bars, sfd=SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model),
            search_strategy=GridStrategy(domain), experiment_dir=directory, checkpoint_interval=1,
            feedback_interval=5, pruning_strategies=reducers)
        loop.run('python-analysis-' + name, n_permutations=30, workers=1, progress_bar=False)
        return loop, directory

    loop, directory = campaign('population', [])
    correlation, importance = analyze_log(loop, directory, space)
    controls = {}
    for name, reducer in metric_reducers().items():
        reduced, location = campaign(name, [reducer])
        controls[name] = {'rows': project.completed_log(reduced).to_dicts(),
            'audit': [json.loads(line) for line in (location / 'audit.jsonl').read_text().splitlines()],
            'checkpoint': json.loads((location / 'checkpoint.json').read_text())['msq_state']}
    return {'rows': project.completed_log(loop).to_dicts(), 'correlation': correlation.to_dicts(),
            'importance': importance.to_dicts(), 'controls': controls}
# endregion analysis_campaign
