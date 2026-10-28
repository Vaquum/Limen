from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess

import numpy as np
import polars as pl
from ruamel.yaml import YAML

from limen import Cohort, Trainer, UniversalExperimentLoop
from limen.backtest import TradeInputs, TradePolicy
from limen.backtest.trade_contract import export_trade_contract
from limen.experiment.holdout import final_holdout_report, freeze_holdout_selection, read_holdout_report, reserve_holdout
from limen.targets import NextBarUpTarget
from limen.yaml import CompiledSFD, build_search_strategy, validate
from limen.yaml.compiler import build_pruning_strategies

import research as research_project
from allocation_research import allocation_case, prefix_probe
from recorded_inputs import FUNDING, HASHES, funding_history, trade_profile

INPUTS = {
    'spot_1h_20240101_20241231.parquet': 'a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72',
    'spot_1h_20240101_20240116.parquet': '170337b8e80843067b3934f194f304bb1e2697327e6784e70739811b3edfdadd',
    'spot_15m_20250101_20250531.parquet': 'eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a',
}


def configuration(name, data_dir):
    config = YAML(typ='safe').load(Path(__file__).with_name('holdout.yaml') if name == 'holdout' else Path(__file__).resolve().parents[2] / 'compiled-research' / (name + '.yaml'))
    params = config['sfd']['manifest']['data_source']['params']
    source = data_dir / ('spot_1h_20240101_20241231.parquet' if name == 'walk_forward' else Path(params['file_path_or_url']).name)
    if hashlib.sha256(source.read_bytes()).hexdigest() != INPUTS[source.name]:
        raise ValueError('Recorded input digest differs: ' + source.name)
    params['file_path_or_url'] = str(source.resolve())
    return config


def execute(config, output, *, bars=None, callback=None, resume=False, project_root=None, factorize=False, post_processing=True):
    report = validate(config)
    if not report.valid:
        raise ValueError(str(report.errors))
    source = Path(config['sfd']['manifest']['data_source']['params']['file_path_or_url'])
    binding = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'selection': None if bars is None else {
        'rows': bars.height, 'sha256': hashlib.sha256(str(bars.schema).encode() + bars.hash_rows(seed=0).to_numpy().tobytes()).hexdigest()}}
    input_record = output / 'project-input.json'
    if resume:
        if json.loads(input_record.read_text()) != binding:
            raise ValueError('Cannot resume this project with changed recorded source rows')
    else:
        output.mkdir(parents=True, exist_ok=True)
        input_record.write_text(json.dumps(binding, sort_keys=True) + '\n')
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=project_root), data=bars, search_strategy=build_search_strategy(config), pruning_strategies=build_pruning_strategies(config), experiment_dir=output, yaml_reference=config, checkpoint_interval=1, feedback_interval=1, intra_callback=callback, project_root=project_root)
    loop.run(config['metadata']['name'], n_permutations=config['uel']['n_permutations'], workers=1, prep_each_round=True, progress_bar=False, post_processing=post_processing, resume=resume, factorize=factorize)
    if 'strict_mode_error' in research_project.completed_log(loop).columns and research_project.completed_log(loop)['strict_mode_error'].drop_nulls().len():
        raise ValueError('A declared scenario round failed its pipeline checks')
    return loop


# region preparation
def preparation(data_dir, output):
    from limen.experiment import MLManifest
    from limen.features import fractional_diff
    from limen.indicators import roc
    from limen.scalers import RobustScaler
    from limen.sfd.reference_architecture import RidgeRegressor
    from limen.targets import TripleBarrierTarget

    raw = pl.read_parquet(configuration('search', data_dir)['sfd']['manifest']['data_source']['params']['file_path_or_url'], n_rows=1440)
    cases = []
    for compressed in (False, True):
        name = 'compressed' if compressed else 'baseline'
        directory = output / name
        directory.mkdir(parents=True)

        def surface(data):
            return data.select('datetime', 'close', 'roc_3', 'roc_12', 'close_fracdiff')

        def capture(data_dict, split_data, round_params, fitted_params):
            for split, frame in zip(('train', 'val', 'test'), split_data, strict=True):
                frame.write_parquet(directory / (split + '.parquet'))
            return data_dict

        manifest = MLManifest()
        (manifest.set_split_config(6, 2, 2)
                    .add_indicator(roc, period=3, group='momentum')
                    .add_indicator(roc, period=12, group='momentum')
                    .add_feature(fractional_diff, d=.4, cols=['close'], threshold=.05, group='price')
                    .add_feature(surface)
                    .with_target_label('label', TripleBarrierTarget, fit_params={
                        'upper_multiple': 1.5, 'lower_multiple': 1.,
                        'max_horizon': 3, 'span': 8, 'min_periods': 8}))
        (manifest.set_scaler(RobustScaler).set_feature_ablation()
                    .set_pca_compression().add_to_data_dict(capture))
        params = {'feature_groups': 'momentum|price', 'feature_drop_count': int(compressed),
                  'feature_drop_seed': 7, 'auto_pca': compressed, 'pca_k': 2}
        data = manifest.prepare_data(raw, params)
        model = RidgeRegressor().train(data, alpha=1., solver='svd')
        scored = model.evaluate(data, inline_metrics=False)
        scaler = data['_scaler']
        pca = data.get('_pca')
        cases.append({'name': name,
                      'features': data.get('_pca_input_feature_names', data['x_train'].columns),
                      'dropped': params.get('_dropped_features', []),
                      'rows': [data['x_' + split].height for split in ('train', 'val', 'test')],
                      'medians': scaler.medians, 'iqrs': scaler.iqrs,
                      'pca_mean': None if pca is None else pca.mean_.tolist(),
                      'components': None if pca is None else pca.components_.tolist(),
                      'predictions': np.asarray(scored['_preds']).tolist(), 'rmse': scored['rmse']})
    return {'cases': cases, 'partition_rows': [864, 288, 288], 'carried_context': True,
            'fixed_d': .4, 'weight_threshold': .05}
# endregion preparation


# region common_baselines
def common_baselines(data_dir, output):
    from limen.experiment.param_domain import ParamDomain
    from limen.experiment.param_search import GridStrategy
    from limen.sfd.foundational_sfd import common_strategies

    source = configuration('search', data_dir)['sfd']['manifest']['data_source']['params']['file_path_or_url']
    raw = pl.read_parquet(source)
    directory = output / 'experiment'
    loop = UniversalExperimentLoop(data=raw, sfd=common_strategies,
                                   search_strategy=GridStrategy(ParamDomain(common_strategies.params())),
                                   experiment_dir=directory)
    loop.run('common-baselines', n_permutations=8, workers=1, record_execution=True,
             recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True),
             post_processing=False, progress_bar=False)
    with gzip.open(directory / 'round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    tracks = {record['round_params']['strategy']: record['execution'] for record in records}
    columns = ['strategy', 'pnl_per_bar_bps_test', 'position_rate_test', 'num_executed_trades_test']
    return {'rows': research_project.completed_log(loop).select(columns).to_dicts(),
            'tracks': tracks, 'source_rows': raw.height, 'fee_bps': 10., 'slip_bps': 5.,
            'execution': 'native lagged long/flat snapshot with per-fill costs'}
# endregion common_baselines


# region search
def search(data_dir, output):
    results = {}
    for kind in ('grid', 'random', 'sobol', 'shuffled', 'factorized'):
        config = configuration('factorized' if kind == 'factorized' else 'search', data_dir)
        config['uel']['search_strategy'] = {'type': kind if kind in ('random', 'sobol') else 'grid'}
        if kind in ('random', 'sobol'):
            config['uel']['search_strategy']['seed'] = 42
        if kind in ('shuffled', 'factorized'):
            config['uel']['search_strategy'].update(shuffle=True, seed=42)
        loop = execute(config, output / kind, factorize=kind == 'factorized')
        results[kind] = research_project.completed_log(loop).select('id', 'accuracy', *(['n_estimators', 'fee'] if kind == 'factorized' else ['C'])).to_dicts()
    return {'runs': results, 'common_baselines': common_baselines(data_dir, output / 'common-baselines'), 'preparation': preparation(data_dir, output / 'preparation'), 'seed': 42, 'families': model_comparisons(data_dir, output / 'families'),
            'adaptive': adaptive_search(data_dir, output / 'adaptive'),
            'feedback': mixed_feedback(data_dir, output / 'feedback')}
# endregion search


# region adaptive_search
def adaptive_search(data_dir, output):
    config = configuration('adaptive', data_dir)
    loop = execute(config, output)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())['msq_state']
    return {'rows': research_project.completed_log(loop).select('id', 'C', 'accuracy').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'checkpoint_yielded': checkpoint['yielded_count'], 'checkpoint_remaining': checkpoint['trim_budget']}
# endregion adaptive_search


# region mixed_feedback
def mixed_feedback(data_dir, output):
    config = configuration('adaptive', data_dir)
    config['uel']['recording'] = {'predictions': True, 'alignment': True, 'trade_ledger': False, 'fold_details': False}

    def steer(log, msq):
        if log.height == 2:
            msq.keep_is('C', .1)
            (output / 'interventions.json').write_text(json.dumps([
                {'op': 'inject_value', 'param': 'C', 'value': 7.5}]))

    loop = execute(config, output, callback=steer)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())
    with gzip.open(output / 'round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    return {'rows': research_project.completed_log(loop).select('id', 'C', 'accuracy').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'domain': checkpoint['domain_state'], 'yielded': checkpoint['msq_state']['yielded_count'],
            'remaining': checkpoint['msq_state']['trim_budget'], 'records': records,
            'predictions': {record['round_id']: recorded_predictions(output, record) for record in records}}
# endregion mixed_feedback


# region recorded_predictions
def recorded_predictions(directory: Path, record):
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata['round_storage'] != 'gzip-jsonl-v2':
        raise ValueError('This project expects its recorded version-2 signal payloads')
    reference = record['preds_ref']
    digest = reference['sha256']
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Invalid recorded prediction digest')
    with gzip.GzipFile(filename=directory / 'round_signals' / (digest + '.json.gz'), mode='rb') as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Recorded prediction payload differs')
    values = json.loads(payload)
    if not isinstance(values, list) or len(values) != reference['count']:
        raise ValueError('Recorded prediction count differs')
    return values
# endregion recorded_predictions


# region model_comparisons
def model_comparisons(data_dir, output):
    families = {}
    for family in ('ridge_regressor', 'dlinear_regressor', 'xgboost_regressor', 'random_binary', 'direction_sizing', 'rbf_svm_binary'):
        config = configuration('family_' + family, data_dir)
        # This diagnostic baseline uses NumPy's caller-owned process RNG.
        if family == 'random_binary':
            np.random.seed(42)
        directory = output / family
        loop = execute(config, directory, post_processing=False)
        log = research_project.completed_log(loop)
        with gzip.open(directory / 'round_data.jsonl.gz', 'rt') as stream:
            records = [json.loads(line) for line in stream]
        metric = 'direction_accuracy' if family == 'direction_sizing' else 'accuracy' if family in ('random_binary', 'rbf_svm_binary') else 'rmse'
        selected = log['id'].head(1).to_list()
        sensors = [] if family == 'random_binary' else Trainer(directory).train(selected)
        predictions = {record['round_id']: recorded_predictions(directory, record) for record in records}
        families[family] = {'metrics': log.select('id', metric).to_dicts(), 'records': records, 'predictions': predictions, 'reconstructed_sensors': len(sensors), 'replay': 'stochastic baseline; replay is not asserted' if family == 'random_binary' else 'selected recorded round reconstructed by Trainer'}
    return families
# endregion model_comparisons


# region recovery
def recovery(data_dir, output):
    config = configuration('recovery', data_dir)
    config['uel']['recording'] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)

    def stop_after_two(log, msq):
        if log.height == 2:
            signal.raise_signal(signal.SIGINT)

    first = execute(config, output / 'experiment', callback=stop_after_two)
    resumed = execute(config, output / 'experiment', resume=True)
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    selected = research_project.completed_log(resumed)['id'][0]
    sensors = Trainer(output / 'experiment').train([selected])
    full = execute(config, output / 'uninterrupted')
    if not research_project.completed_log(resumed).drop('execution_time').equals(research_project.completed_log(full).drop('execution_time')):
        raise ValueError('Resumed Compiled results differ from uninterrupted research')
    return {'interrupted_rounds': research_project.completed_log(first).height, 'rounds': research_project.completed_log(resumed).height, 'round_indices': [row['_round_index'] for row in records], 'round_ids': [row['round_id'] for row in records], 'reconstructed_sensors': len(sensors), 'uninterrupted_equivalence': True}
# endregion recovery


def fold_metadata(sensor):
    fold = getattr(sensor, 'fold')
    fitting = getattr(sensor, 'fitting_cutoff')
    selection = getattr(sensor, 'selection_cutoff')
    if not isinstance(fold, int) or not isinstance(fitting, datetime) or not isinstance(selection, datetime):
        raise ValueError('Trainer must provide a qualified fold and its fitting and selection cutoffs')
    return {'fold': fold, 'fitting': fitting.isoformat(), 'selection': selection.isoformat()}


# region acceptance
def acceptance(data_dir, output):
    config = configuration('acceptance', data_dir)
    loop = execute(config, output)
    report = json.loads((output / 'acceptance_report.json').read_text())
    return {'rounds': research_project.completed_log(loop).height, 'report': report,
            'artifacts': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in (
                'trial_returns.parquet', 'acceptance_report.json', 'acceptance_report.md', 'round_data.jsonl.gz')}}
# endregion acceptance


# region walk_forward
def walk_forward(data_dir, output):
    config = configuration('walk_forward', data_dir)
    bars = pl.read_parquet(config['sfd']['manifest']['data_source']['params']['file_path_or_url'], n_rows=384)
    loop = execute(config, output / 'experiment', bars=bars)
    round_id = research_project.completed_log(loop)['id'][0]
    trainer = Trainer(output / 'experiment', data=bars)
    sensors = [trainer.train([round_id], fold=index)[0] for index in range(2)]
    cohort = Cohort(rounds=[(output / 'experiment', round_id, index) for index in range(2)])
    cohort.set_members(sensors)
    metadata = [fold_metadata(sensor) for sensor in sensors]
    round_records = output / 'experiment/round_data.jsonl.gz'
    return {'acceptance': acceptance(data_dir, output / 'acceptance'),
            'rounds': research_project.completed_log(loop).height, 'folds': loop.fold_results.select('fold', 'id', 'backtest_total_return', 'val_backtest_total_return', 'val_score', 'optimal_threshold').to_dicts(),
            'objective': research_project.completed_log(loop)['val_backtest_total_return'][0],
            'round_records': {'path': 'experiment/round_data.jsonl.gz', 'sha256': hashlib.sha256(round_records.read_bytes()).hexdigest()},
            'member_folds': [row['fold'] for row in metadata], 'fitting_cutoffs': [row['fitting'] for row in metadata], 'selection_cutoffs': [row['selection'] for row in metadata], 'member_ids': cohort.member_ids, 'origin': 'qualified recorded folds'}
# endregion walk_forward


# region holdout
def holdout(data_dir, output):
    project = output / 'project'
    project.mkdir(parents=True)
    (project / 'limen.toml').write_text('[project]\nname="book-final-observations"\n')
    subprocess.run(['git', 'init', '--quiet', str(project)], check=True, timeout=10)
    reservation = reserve_holdout(project, '2024-01-05', '2024-01-06', interval_seconds=3600)
    config = configuration('holdout', data_dir)
    directory = project / 'experiment'
    loop = execute(config, directory, project_root=project)
    selected_ids = research_project.completed_log(loop)['id'].head(1).to_list()
    selected = Trainer(directory, project_root=project).train(selected_ids)
    cohort = Cohort(experiment_log_path=str(directory), permutation_ids=selected_ids)
    cohort.set_members(selected)
    frozen = freeze_holdout_selection(cohort, context_start='2024-01-04', scenarios=[dict(fee_bps=5.0, slip_bps=2.0, notional_rate=.25, execution_lag_bars=1, take_profit_bps=None, stop_loss_bps=None)], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'))
    report = final_holdout_report(frozen, project_root=project, reservation_id=reservation)
    attempt_id = report['attempt_id']
    if not isinstance(attempt_id, str):
        raise ValueError('The final report must have an attempt identity')
    if read_holdout_report(project, attempt_id) != report:
        raise ValueError('Retained final report differs')
    (output / 'retained-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return {'rounds': research_project.completed_log(loop).height, 'planned': report['planned'], 'classification': report['classification'], 'scenarios': report['scenarios'], 'exposure': report['exposure'], 'rows_sha256': report['rows_sha256'], 'source': report['source']}
# endregion holdout


# region calibration
def calibration(data_dir, output):
    config = configuration('calibration', data_dir)
    loop = execute(config, output / 'experiment')
    sensors = Trainer(output / 'experiment').train(research_project.completed_log(loop)['id'].to_list())
    return {'modes': research_project.completed_log(loop).select('use_calibration', 'use_threshold', 'optimal_threshold', 'val_score', 'accuracy').to_dicts(), 'reconstructed_sensors': len(sensors), 'validation_reused': True}
# endregion calibration


# region execution
def execution_contexts(config, loop):
    manifest = CompiledSFD(config).manifest()
    if not isinstance(loop.data, pl.DataFrame):
        raise TypeError('Execution must retain its original recorded source')
    contexts = []
    for row in research_project.completed_log(loop).iter_rows(named=True):
        prepared = manifest.prepare_data(loop.data, {key: row[key] for key in config['sfd']['params']})
        inputs, policy = prepared['_trade_inputs'], prepared['_trade_policy']
        if not isinstance(inputs, TradeInputs) or not isinstance(policy, TradePolicy):
            raise TypeError('Execution requires native bound inputs and policy')
        contexts.append({'observations': inputs.observations.to_dicts(), 'partition': [inputs.partition_start_ns, inputs.partition_end_ns],
                         'contract': export_trade_contract(policy, inputs)})
    return contexts


def execution(data_dir, output):
    config = configuration('execution', data_dir)
    loop = execute(config, output / 'experiment')
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    history = funding_history(data_dir)
    derived = output / 'recorded-funding.parquet'
    history.write_parquet(derived)
    funded_config = configuration('funded', data_dir)
    funded_config['sfd']['manifest']['backtest']['funding']['data_source']['params']['file_path_or_url'] = str(derived.resolve())
    funded_loop = execute(funded_config, output / 'funded')
    with gzip.open(output / 'funded/round_data.jsonl.gz', 'rt') as stream:
        funded_records = [json.loads(line) for line in stream]
    funded = {'metrics': research_project.completed_log(funded_loop).select('risk_span', 'backtest_funding_pnl').to_dicts(), 'ledgers': [row['trade_ledger'] for row in funded_records], 'contexts': execution_contexts(funded_config, funded_loop), 'recorded_events': history.height, 'input_sha256': HASHES[FUNDING]}
    return {'rounds': research_project.completed_log(loop).height, 'metrics': research_project.completed_log(loop).select('risk_span', 'backtest_total_return', 'backtest_completed_trades').to_dicts(), 'contracts': [row['trade_contract_digest'] for row in records], 'ledgers': [row['trade_ledger'] for row in records], 'contexts': execution_contexts(config, loop), 'funded': funded}
# endregion execution


# region bars
def bars(data_dir, output):
    config = configuration('bars', data_dir)
    loop = execute(config, output / 'experiment')
    return {'rounds': research_project.completed_log(loop).height, 'modes': research_project.completed_log(loop).select('bar_type', 'accuracy').to_dicts(), 'placement': 'bar formation runs separately after each chronological split', 'trade_profile': trade_profile(data_dir, output)}
# endregion bars


# region allocation
def allocation(data_dir, output):
    config = configuration('allocation', data_dir)
    loop = execute(config, output / 'experiment')
    columns = ['risk_span', 'allocation_training_completed', 'allocation_training_unavailable', 'backtest_total_return', 'backtest_completed_trades']
    cases = {}
    for name in ('rule', 'ml', 'funded', 'unfunded'):
        config = configuration('allocation_ml' if name == 'ml' else 'allocation_funded' if name in ('funded', 'unfunded') else 'allocation', data_dir)
        if name in ('funded', 'unfunded'):
            history = funding_history(data_dir)
            source = output / 'recorded-funding.parquet'
            history.write_parquet(source)
            backtest = config['sfd']['manifest']['backtest']
            backtest['funding']['data_source']['params']['file_path_or_url'] = str(source.resolve())
            if name == 'unfunded':
                del backtest['funding']
        report, model, primary = allocation_case(config, output / name)
        if name == 'ml':
            params = {key: values[0] for key, values in config['sfd']['params'].items()}
            report['prefix'] = prefix_probe(model, primary, params)
        cases[name] = report
    return {'rounds': research_project.completed_log(loop).height, 'results': research_project.completed_log(loop).select(columns).to_dicts(),
            'cases': cases, 'promotion': 'batch research only; no live Sensor promotion or resumed allocation claim'}
# endregion allocation


# region command_line
def command_line(data_dir, output):
    from command_line import run as run_commands
    return run_commands(configuration('recovery', data_dir), output)
# endregion command_line


SCENARIOS = {'search': search, 'recovery': recovery, 'walk_forward': walk_forward, 'holdout': holdout, 'calibration': calibration, 'execution': execution, 'bars': bars, 'allocation': allocation, 'command_line': command_line}


def run(name, data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    result = SCENARIOS[name](data_dir, output)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
