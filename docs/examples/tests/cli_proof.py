from datetime import datetime
import subprocess
import gzip
import hashlib
import json

import numpy as np
import polars as pl
import pytest

from scenario_oracles import verify_acceptance


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
    with gzip.open(directory / 'round_signals' / (reference['sha256'] + '.json.gz'), 'rb') as stream:
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
    for row, record in zip(log.iter_rows(named=True), records(directory), strict=True):
        ledger = record['trade_ledger']
        fills = ledger['fills']
        assert fills and row['backtest_completed_trades'] > 0
        assert all(fill['time_ns'] >= fill['available_at_ns'] for fill in fills if 'available_at_ns' in fill)
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
    allocation = {name: checked_log(base / ('compiled-' + name), 2).height for name in ('allocation', 'allocation_ml', 'allocation_funded')}
    summaries['allocation'] = allocation
    original, restored = project / 'results/compiled-project', project / 'results/compiled-restored'
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=original, text=True).split('\0')
    assert all((original / path).read_bytes() == (restored / path).read_bytes() for path in tracked if path and not (original / path).is_symlink())
    summaries['command_line'] = {'restored_files': len([name for name in tracked if name])}
    inventory = {}
    for scenario, directories in DIRECTORIES.items():
        inventory[scenario] = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest() for name in directories for path in (base / name).rglob('*') if path.is_file()}
        assert inventory[scenario]
    inventory['command_line'] = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest() for path in original.rglob('*.yaml')}
    (project / 'results/compiled-artifacts.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (project / 'results/compiled').mkdir(exist_ok=True)
    (project / 'results/compiled/summary.json').write_text(json.dumps(summaries['selection'], indent=2) + '\n')
    (project / 'results/compiled-all').mkdir(exist_ok=True)
    (project / 'results/compiled-all/summary.json').write_text(json.dumps({'scenarios': {key: value for key, value in summaries.items() if key != 'selection'}}, indent=2, allow_nan=False) + '\n')
    return summaries
