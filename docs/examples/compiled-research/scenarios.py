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
    'spot_15m_20250101_20250531.parquet': 'eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a',
}


def configuration(name, data_dir):
    config = YAML(typ='safe').load(Path(__file__).with_name(name + '.yaml'))
    params = config['sfd']['manifest']['data_source']['params']
    source = data_dir / Path(params['file_path_or_url']).name
    if hashlib.sha256(source.read_bytes()).hexdigest() != INPUTS[source.name]:
        raise ValueError('Recorded input digest differs: ' + source.name)
    params['file_path_or_url'] = str(source.resolve())
    return config


def execute(config, output, *, bars=None, callback=None, resume=False, project_root=None, factorize=False, post_processing=True):
    report = validate(config)
    if not report.valid:
        raise ValueError(str(report.errors))
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=project_root), data=bars, search_strategy=build_search_strategy(config), pruning_strategies=build_pruning_strategies(config), experiment_dir=output, yaml_reference=config, checkpoint_interval=1, feedback_interval=1, intra_callback=callback, project_root=project_root)
    loop.run(config['metadata']['name'], n_permutations=config['uel']['n_permutations'], workers=1, prep_each_round=True, progress_bar=False, post_processing=post_processing, resume=resume, factorize=factorize)
    if 'strict_mode_error' in research_project.completed_log(loop).columns and research_project.completed_log(loop)['strict_mode_error'].drop_nulls().len():
        raise ValueError('A declared scenario round failed its pipeline checks')
    return loop


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
    return {'runs': results, 'seed': 42, 'families': model_comparisons(data_dir, output / 'families'), 'adaptive': adaptive_search(data_dir, output / 'adaptive')}
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


# region model_comparisons
def model_comparisons(data_dir, output):
    families = {}
    for family in ('ridge_regressor', 'dlinear_regressor', 'xgboost_regressor', 'random_binary', 'direction_sizing'):
        config = configuration('family_' + family, data_dir)
        # This diagnostic baseline uses NumPy's caller-owned process RNG.
        if family == 'random_binary':
            np.random.seed(42)
        directory = output / family
        loop = execute(config, directory, post_processing=False)
        log = research_project.completed_log(loop)
        with gzip.open(directory / 'round_data.jsonl.gz', 'rt') as stream:
            records = [json.loads(line) for line in stream]
        metric = 'direction_accuracy' if family == 'direction_sizing' else 'accuracy' if family == 'random_binary' else 'rmse'
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
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    return {'rounds': research_project.completed_log(loop).height, 'folds': loop.fold_results.select('fold', 'id', 'backtest_total_return', 'val_backtest_total_return', 'val_score', 'optimal_threshold').to_dicts(),
            'objective': research_project.completed_log(loop)['val_backtest_total_return'][0], 'records': records,
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
