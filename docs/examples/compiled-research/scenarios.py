import gzip
import hashlib
import json
from pathlib import Path
import signal
import subprocess

import polars as pl
from ruamel.yaml import YAML

from limen import Cohort, Trainer, UniversalExperimentLoop
from limen.experiment.holdout import final_holdout_report, freeze_holdout_selection, read_holdout_report, reserve_holdout
from limen.targets import NextBarUpTarget
from limen.yaml import CompiledSFD, build_search_strategy, validate

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


def execute(config, output, *, bars=None, callback=None, resume=False, project_root=None, factorize=False):
    report = validate(config)
    if not report.valid:
        raise ValueError(str(report.errors))
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=project_root), data=bars, search_strategy=build_search_strategy(config), experiment_dir=output, yaml_reference=config, checkpoint_interval=1, feedback_interval=1, intra_callback=callback, project_root=project_root)
    loop.run(config['metadata']['name'], n_permutations=config['uel']['n_permutations'], workers=1, prep_each_round=True, progress_bar=False, post_processing=True, resume=resume, factorize=factorize)
    if 'strict_mode_error' in loop.experiment_log.columns and loop.experiment_log['strict_mode_error'].drop_nulls().len():
        raise ValueError('A declared scenario round failed its pipeline checks')
    return loop


# region search
def search(data_dir, output):
    results = {}
    for kind in ('grid', 'random', 'sobol', 'shuffled', 'factorized'):
        config = configuration('search', data_dir)
        config['uel']['search_strategy'] = {'type': kind if kind in ('random', 'sobol') else 'grid'}
        if kind in ('random', 'sobol'):
            config['uel']['search_strategy']['seed'] = 42
        if kind in ('shuffled', 'factorized'):
            config['uel']['search_strategy'].update(shuffle=True, seed=42)
        loop = execute(config, output / kind, factorize=kind == 'factorized')
        results[kind] = loop.experiment_log.select('id', 'C', 'accuracy').to_dicts()
    return {'runs': results, 'seed': 42}
# endregion search


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
    selected = resumed.experiment_log['id'][0]
    sensors = Trainer(output / 'experiment').train([selected])
    return {'interrupted_rounds': first.experiment_log.height, 'rounds': resumed.experiment_log.height, 'round_indices': [row['_round_index'] for row in records], 'round_ids': [row['round_id'] for row in records], 'reconstructed_sensors': len(sensors)}
# endregion recovery


# region walk_forward
def walk_forward(data_dir, output):
    config = configuration('walk_forward', data_dir)
    bars = pl.read_parquet(config['sfd']['manifest']['data_source']['params']['file_path_or_url'], n_rows=384)
    loop = execute(config, output / 'experiment', bars=bars)
    round_id = loop.experiment_log['id'][0]
    trainer = Trainer(output / 'experiment', data=bars)
    sensors = [trainer.train([round_id], fold=index)[0] for index in range(2)]
    cohort = Cohort(rounds=[(output / 'experiment', round_id, index) for index in range(2)])
    cohort.set_members(sensors)
    return {'rounds': loop.experiment_log.height, 'folds': loop.fold_results.to_dicts(), 'member_folds': [sensor.fold for sensor in sensors], 'fitting_cutoffs': [sensor.fitting_cutoff.isoformat() for sensor in sensors], 'selection_cutoffs': [sensor.selection_cutoff.isoformat() for sensor in sensors], 'member_ids': cohort.member_ids, 'origin': 'qualified recorded folds'}
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
    selected = Trainer(directory, project_root=project).train([loop.experiment_log['id'][0]])
    cohort = Cohort(experiment_log_path=str(directory), permutation_ids=[sensor.permutation_id for sensor in selected])
    cohort.set_members(selected)
    frozen = freeze_holdout_selection(cohort, context_start='2024-01-04', scenarios=[dict(fee_bps=5.0, slip_bps=2.0, notional_rate=.25, execution_lag_bars=1, take_profit_bps=None, stop_loss_bps=None)], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'))
    report = final_holdout_report(frozen, project_root=project, reservation_id=reservation)
    if read_holdout_report(project, report['attempt_id']) != report:
        raise ValueError('Retained final report differs')
    (output / 'retained-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return {'rounds': loop.experiment_log.height, 'planned': report['planned'], 'classification': report['classification'], 'scenarios': report['scenarios'], 'exposure': report['exposure'], 'rows_sha256': report['rows_sha256'], 'source': report['source']}
# endregion holdout


# region calibration
def calibration(data_dir, output):
    config = configuration('calibration', data_dir)
    loop = execute(config, output / 'experiment')
    sensors = Trainer(output / 'experiment').train(loop.experiment_log['id'].to_list())
    return {'modes': loop.experiment_log.select('use_calibration', 'use_threshold', 'optimal_threshold', 'val_score', 'accuracy').to_dicts(), 'reconstructed_sensors': len(sensors), 'validation_reused': True}
# endregion calibration


# region execution
def execution(data_dir, output):
    config = configuration('execution', data_dir)
    loop = execute(config, output / 'experiment')
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    return {'rounds': loop.experiment_log.height, 'metrics': loop.experiment_log.select('risk_span', 'backtest_total_return', 'backtest_completed_trades').to_dicts(), 'contracts': [row['trade_contract_digest'] for row in records], 'ledgers': [row['trade_ledger'] for row in records]}
# endregion execution


# region bars
def bars(data_dir, output):
    config = configuration('bars', data_dir)
    loop = execute(config, output / 'experiment')
    return {'rounds': loop.experiment_log.height, 'modes': loop.experiment_log.select('bar_type', 'accuracy').to_dicts(), 'placement': 'bar formation runs separately after each chronological split'}
# endregion bars


# region allocation
def allocation(data_dir, output):
    config = configuration('allocation', data_dir)
    loop = execute(config, output / 'experiment')
    columns = [name for name in loop.experiment_log.columns if name.startswith(('allocation_', 'backtest_')) or name == 'risk_span']
    return {'rounds': loop.experiment_log.height, 'results': loop.experiment_log.select(columns).to_dicts(), 'promotion': 'batch research only; no live Sensor promotion or resumed allocation claim'}
# endregion allocation


SCENARIOS = {'search': search, 'recovery': recovery, 'walk_forward': walk_forward, 'holdout': holdout, 'calibration': calibration, 'execution': execution, 'bars': bars, 'allocation': allocation}


def run(name, data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    return SCENARIOS[name](data_dir, output)
