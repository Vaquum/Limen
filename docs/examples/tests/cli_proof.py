from datetime import datetime
import subprocess
import gzip
import hashlib
import json

import numpy as np
import polars as pl
import pytest

from scenario_oracles import execution_clocks, verify_acceptance


DIRECTORIES = {
    'selection': ['compiled-cli'],
    'search': ['compiled-research', 'compiled-preparation', 'compiled-search', 'compiled-search_random', 'compiled-search_sobol', 'compiled-factorized', 'compiled-adaptive', 'compiled-common_strategies', *['compiled-family_' + name for name in ('ridge_regressor', 'dlinear_regressor', 'xgboost_regressor', 'random_binary', 'direction_sizing')]],
    'recovery': ['compiled-recovery'], 'walk_forward': ['compiled-walk_forward', 'compiled-acceptance'],
    'holdout': ['compiled-holdout'], 'calibration': ['compiled-calibration'],
    'execution': ['compiled-execution', 'compiled-funded'], 'bars': ['compiled-bars'],
    'allocation': ['compiled-allocation', 'compiled-allocation_ml', 'compiled-allocation_funded'],
}


def records(directory):
    with gzip.open(directory / 'round_data.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def checked_log(directory, count):
    log = pl.read_csv(directory / 'results.csv')
    assert log.height == log['id'].n_unique() == count, directory
    if 'strict_mode_error' in log.columns:
        assert log['strict_mode_error'].drop_nulls().len() == 0
    assert len(records(directory)) == count
    return log


def predictions(directory, record):
    reference = record['preds_ref']
    with gzip.GzipFile(filename=directory / 'round_signals' / (reference['sha256'] + '.json.gz'), mode='rb') as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == reference['sha256']
    result = np.asarray(json.loads(raw))
    assert len(result) == reference['count'] and np.isfinite(result).all()
    return result


def verify_regressors(project):
    raw = pl.read_parquet(project / 'inputs/spot_15m_20250101_20250531.parquet')
    labels = raw.with_columns(((pl.col('close').shift(-1) / pl.col('close') - 1) * 100).alias('label'))
    result = {}
    for family in ('ridge_regressor', 'dlinear_regressor', 'xgboost_regressor'):
        directory = project / 'results/dev' / ('compiled-family_' + family)
        log = checked_log(directory, 1)
        record = records(directory)[0]
        values = predictions(directory, record)
        alignment = record['alignment']
        truth = labels.filter(pl.col('datetime').is_between(datetime.fromisoformat(alignment['first_test_datetime']), datetime.fromisoformat(alignment['last_test_datetime'])))['label'].to_numpy()
        assert len(values) == len(truth) > 0
        rmse = float(np.sqrt(np.mean((values - truth) ** 2)))
        assert log['rmse'][0] == pytest.approx(rmse, rel=1e-10, abs=1e-12)
        result[family] = {'rmse': rmse, 'rows': len(values)}
    return result


def verify_execution(project, name):
    directory = project / 'results/dev' / ('compiled-' + name)
    log = checked_log(directory, 2)
    assert set(log['risk_span']) == {6, 12}
    raw_funding = json.loads((project / 'inputs/btcusdt_funding_20240101_20240108.json').read_text())['records']
    rates = {row['fundingTime'] * 1_000_000: (float(row['fundingRate']), float(row['markPrice'])) for row in raw_funding}
    raw = pl.read_parquet(project / 'inputs/spot_1h_20240101_20241231.parquet')
    for row, record in zip(log.iter_rows(named=True), records(directory), strict=True):
        ledger = record['trade_ledger']
        fills = ledger['fills']
        assert fills and row['backtest_completed_trades'] > 0
        contract = record['trade_contract']
        partition = contract['partition']
        interval = contract['policy']['source_interval_seconds'] * 1_000_000_000
        source = raw.filter(pl.col('datetime').dt.epoch('ns').is_between(*partition, closed='left'))
        observations = [dict(item, row_id=f'row:{index}', start_ns=stamp, end_ns=stamp + interval,
                             open_available_at_ns=stamp, available_at_ns=stamp + interval)
                        for index, (item, stamp) in enumerate(zip(source.to_dicts(), source['datetime'].dt.epoch('ns'), strict=True))]
        execution_clocks({'observations': observations, 'contract': contract, 'partition': partition}, ledger, raw)
        final = ledger['states'][-1]
        assert row['backtest_total_return'] == pytest.approx(final['equity'] / contract['initial_conditions']['equity'] - 1)
        assert row['backtest_fees'] == pytest.approx(sum(fill['fee'] for fill in fills))
        if name == 'funded':
            assert ledger['funding']
            for payment in ledger['funding']:
                rate, price = rates[payment['end_ns']]
                quantity = sum(fill['quantity_delta'] for fill in fills if fill['time_ns'] < payment['end_ns'])
                assert payment['quantity'] == pytest.approx(quantity, rel=1e-12, abs=1e-12)
                assert (payment['rate_decimal'], payment['valuation_price']) == (rate, price)
                assert payment['recognized_delta'] == pytest.approx(-quantity * rate * price, rel=1e-12, abs=1e-12)
            assert row['backtest_funding_pnl'] == pytest.approx(sum(payment['recognized_delta'] for payment in ledger['funding']))
    return {'rounds': log.height, 'risk_spans': log['risk_span'].to_list()}


def verify_allocation(project, name):
    directory = project / 'results/dev' / ('compiled-' + name)
    log = checked_log(directory, 2)
    metadata = json.loads((directory / 'metadata.json').read_text())
    dates = metadata['yaml_reference']['sfd']['manifest']['split_dates']
    raw = pl.read_parquet(project / 'inputs/spot_1h_20240101_20241231.parquet')
    window = [datetime.fromisoformat(dates[key]).replace(tzinfo=raw['datetime'][0].tzinfo) for key in ('train_start', 'train_end')]
    training = raw.filter(pl.col('datetime').is_between(*window, closed='left'))
    raw_digest = hashlib.sha256(str(training.schema).encode() + training.hash_rows(seed=0).to_numpy().tobytes()).hexdigest()
    for row, record in zip(log.iter_rows(named=True), records(directory), strict=True):
        model = record['learning_binding']['model']
        assert model['training']['raw_train'] == raw_digest
        assert model['primary']['reference'].endswith('.logreg_binary' if name == 'allocation_ml' else '.rule_based')
        outcomes = pl.DataFrame(record['trade_ledger']['allocation_training'])
        complete = outcomes.filter(pl.col('available'))
        assert complete.height == row['allocation_training_completed'] >= 2
        assert outcomes['opportunity_id'].n_unique() == outcomes.height
        np.testing.assert_allclose(complete['net_return_on_entry_notional'], complete['net_pnl'] / complete['entry_notional'])
        for fold in model['training']['folds']:
            samples = complete.filter(pl.col('opening_block') == fold['block'])
            assert samples['label_available_ns'].lt(fold['end_ns']).all()
            assert samples['decision_available_at_ns'].ge(fold['start_ns']).all()
            if name == 'allocation_ml':
                assert 0 < fold['fit_prefix_rows'] < training.height
        exposure = predictions(directory, record)
        assert (exposure >= 0).all() and (exposure <= model['allocation_params']['max_size']).all()
        if name == 'allocation_funded':
            policy = model['training']['economics']['funding']
            assert policy['preset_id'] == 'binance_btcusdt' and policy['approximation'] == 'recorded_exact'
            assert record['trade_ledger']['funding']
    return log.height


def verify_project(project):
    base = project / 'results/dev'
    summaries = {}
    selection = checked_log(base / 'compiled-cli', 2)
    assessment = json.loads((base / 'compiled-cli/assessment.json').read_text())
    expected = selection.sort(['val_backtest_total_return', 'id'], descending=[True, False])['id'][0]
    assert assessment['selected_ids'] == [expected] and assessment['members'] == 1
    tape = pl.read_parquet(base / 'compiled-cli/predictions.parquet')
    assert tape.filter(pl.col('reason').is_null()).height > 0
    summaries['selection'] = {'interface': 'compiled', 'scenario': 'selection', 'rounds': 2, 'selected_rounds': 1, 'reconstructed_sensors': 1}
    main = checked_log(base / 'compiled-research', 8)
    assert set(main['auto_pca']) == {True, False} and set(main['feature_drop_count']) == {0, 1}
    selected = json.loads((base / 'compiled-research/assessment.json').read_text())
    assert selected['selected_ids'] == main.sort(['val_backtest_total_return', 'id'], descending=[True, False])['id'].head(2).to_list()
    assert selected['members'] == 2
    domains = []
    for name in ('search', 'search_random', 'search_sobol'):
        log = checked_log(base / ('compiled-' + name), 4)
        domains.append(dict(zip(log['id'], log['accuracy'], strict=True)))
    assert domains[0] == domains[1] == domains[2]
    factorized = checked_log(base / 'compiled-factorized', 4)
    assert set(zip(factorized['fee'], factorized['n_estimators'])) == {(fee, trees) for fee in (7., 9.) for trees in (4, 8)}
    adaptive = checked_log(base / 'compiled-adaptive', 3)
    audit = [json.loads(line) for line in (base / 'compiled-adaptive/audit.jsonl').read_text().splitlines()]
    assert any(item.get('source') == 'BudgetReducer' for cycle in audit for item in cycle['interventions'])
    checked_log(base / 'compiled-preparation', 4)
    checked_log(base / 'compiled-common_strategies', 8)
    checked_log(base / 'compiled-family_random_binary', 1)
    signed = checked_log(base / 'compiled-family_direction_sizing', 2)
    assert signed.height == 2
    summaries['search'] = {'rounds': main.height, 'selected_members': selected['members'], 'traversals': len(domains), 'budget_rounds': adaptive.height, 'families': verify_regressors(project)}
    recovery = checked_log(base / 'compiled-recovery', 4)
    assert [row['_round_index'] for row in records(base / 'compiled-recovery')] == list(range(4))
    summaries['recovery'] = {'rounds': recovery.height, 'unique_ids': recovery['id'].n_unique()}
    checked_log(base / 'compiled-walk_forward', 1)
    folds = records(base / 'compiled-walk_forward')[0]['folds']
    assert len(folds) == 2
    membership = json.loads((base / 'compiled-walk_forward/assessment.json').read_text())
    assert membership['folds'] == [0, 1] and membership['members'] == 2
    acceptance_dir = base / 'compiled-acceptance'
    acceptance = {'rounds': 2, 'report': json.loads((acceptance_dir / 'acceptance_report.json').read_text()), 'artifacts': {name: hashlib.sha256((acceptance_dir / name).read_bytes()).hexdigest() for name in ('trial_returns.parquet', 'acceptance_report.json', 'acceptance_report.md', 'round_data.jsonl.gz')}}
    verify_acceptance(acceptance, acceptance_dir, project / 'inputs')
    summaries['walk_forward'] = {'folds': len(folds), 'members': membership['members'], 'acceptance': acceptance['report']}
    report = json.loads((base / 'compiled-holdout/final-report.json').read_text())
    rows_path = project / 'compiled-research/final' / report['rows_path']
    assert hashlib.sha256(rows_path.read_bytes()).hexdigest() == report['rows_sha256']
    assert pl.read_parquet(rows_path).filter(pl.col('role') == 'final').height == 24
    assert report['classification']['accuracy']['denominator'] == 23
    assert report['exposure']['class'] == 'unknown'
    summaries['holdout'] = {'final_rows': 24, 'report_sha256': hashlib.sha256((base / 'compiled-holdout/final-report.json').read_bytes()).hexdigest(), 'exposure': report['exposure']}
    calibration = checked_log(base / 'compiled-calibration', 4)
    assert set(zip(calibration['use_calibration'], calibration['use_threshold'])) == {(a, b) for a in (False, True) for b in (False, True)}
    summaries['calibration'] = {'modes': calibration.select('use_calibration', 'use_threshold', 'optimal_threshold').to_dicts()}
    summaries['execution'] = {name: verify_execution(project, name) for name in ('execution', 'funded')}
    bars = checked_log(base / 'compiled-bars', 3)
    assert set(bars['bar_type']) == {'volume', 'trade', 'liquidity'}
    summaries['bars'] = {'modes': bars['bar_type'].to_list()}
    allocation = {name: verify_allocation(project, name) for name in ('allocation', 'allocation_ml', 'allocation_funded')}
    summaries['allocation'] = allocation
    original, restored = project / 'results/compiled-project', project / 'results/compiled-restored'
    index = json.loads((original / 'manifests/committed/index.json').read_text())['manifests']
    assert len(index) == 2
    parent = next(item for item in index if item['parent_id'] is None)
    child = next(item for item in index if item['parent_id'] is not None)
    assert child['parent_id'] == parent['id']
    runs = list(original.glob('results/*/*/metadata.json'))
    assert len(runs) == 1 and json.loads(runs[0].read_text())['manifest_id'] == parent['id']
    campaign = pl.read_csv(runs[0].parent / 'results.csv')
    parquet = pl.read_parquet(runs[0].parent / 'results.parquet')
    assert campaign.height == campaign['id'].n_unique() == 4
    assert set(campaign['C']) == {.1, .5, 1., 2.}
    assert campaign.columns == parquet.columns
    for column in campaign.columns:
        if campaign[column].dtype.is_numeric():
            np.testing.assert_allclose(campaign[column], parquet[column], equal_nan=True)
        else:
            assert campaign[column].to_list() == parquet[column].to_list()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=original, text=True).split('\0')
    assert all((original / path).read_bytes() == (restored / path).read_bytes() for path in tracked if path and not (original / path).is_symlink())
    summaries['command_line'] = {'restored_files': len([name for name in tracked if name])}
    inventory = {}
    for scenario, directories in DIRECTORIES.items():
        inventory[scenario] = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest() for name in directories for path in (base / name).rglob('*') if path.is_file()}
        assert inventory[scenario]
    inventory['command_line'] = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest() for path in original.rglob('*.yaml')}
    (project / 'results/compiled-artifacts.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return summaries


def prove_interrupted_recovery(project, environment):
    import shutil
    import signal
    import time
    directory = project / 'results/compiled-interruption'
    (directory / 'compiled-research').mkdir(parents=True)
    (directory / 'inputs').mkdir()
    shutil.copyfile(project / 'compiled-research/recovery.yaml', directory / 'compiled-research/recovery.yaml')
    shutil.copyfile(project / 'inputs/spot_1h_20240101_20241231.parquet', directory / 'inputs/spot_1h_20240101_20241231.parquet')
    child_environment = dict(environment)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        child_environment.pop(name, None)
    child_environment['PYTHONNOUSERSITE'] = '1'
    process = subprocess.Popen(['limen', 'run', '--no-progress-bar', 'compiled-research/recovery.yaml'], cwd=directory, env=child_environment)
    checkpoint = directory / 'results/dev/compiled-recovery/checkpoint.json'
    deadline, interrupted = time.monotonic() + 30, False
    while process.poll() is None and time.monotonic() < deadline:
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())['msq_state']
            if 0 < state['yielded_count'] < 4:
                process.send_signal(signal.SIGINT)
                interrupted = True
                break
        time.sleep(.001)
    if process.poll() is None and not interrupted:
        process.terminate()
    _ = process.wait(timeout=10)
    assert interrupted, 'The actual CLI process did not expose an interruptible completed-round checkpoint'
    output = directory / 'results/dev/compiled-recovery'
    first = pl.read_csv(output / 'results.csv')
    assert 0 < first.height < 4
    subprocess.run(['limen', 'run', '--no-progress-bar', '--resume', 'results/dev/compiled-recovery'], cwd=directory, env=child_environment, check=True, timeout=30)
    complete = checked_log(output, 4)
    baseline = pl.read_csv(project / 'results/dev/compiled-recovery/results.csv')
    assert complete.drop('execution_time').equals(baseline.drop('execution_time'))
    assert set(first['id']) <= set(complete['id'])
    assert [row['_round_index'] for row in records(output)] == list(range(4))
    return {'interrupted_rounds': first.height, 'resumed_rounds': complete.height, 'uninterrupted_equivalence': True}
