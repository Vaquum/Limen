import copy
import csv
import json
import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from limen import __version__
from limen.data import HistoricalData
from limen.experiment import RuleBasedManifest, UniversalExperimentLoop
from limen.experiment import _round_storage as _storage
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.sfd.reference_architecture.rule_based import rule_based
from limen.sfd.rule_based.config import RuleBasedConfig
from limen.sfd.rule_based.predicates import build_predicate, with_persistence, with_recency
from limen.yaml import CompiledSFD, build_search_strategy, validate


FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
DATES = ('2024-01-01', '2024-01-03', '2024-01-03', '2024-01-05',
         '2024-01-05', '2024-01-07')
DATE_KEYS = ('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end')
FIELDS = ('persistence_n', 'recency_n')
INVALID = (0, -1, True, False, np.bool_(True), 1.0, 1.5, '3', None,
           float('nan'), float('inf'), [3], {'window': 3})
BAD_REFERENCES = ('window', '', '{}', '{window:.0f}', '{window!s}', '{window.real}',
                  '{window[0]}', 'prefix_{window}', '{window}{other}', '{{window}}', '{window')


@pytest.fixture(scope='module')
def recorded_temporal_bars():
    return pl.read_parquet(FIXTURE, n_rows=168)


def _conditions(persistence='{hold_bars}', recency='{recent_bars}'):
    return [
        {'id': 'up', 'name': 'Persistent up bar', 'type': 'relative', 'column': 'close',
         'operator': '>', 'other_column': 'open', 'persistence_n': persistence},
        {'id': 'down', 'name': 'Recent down bar', 'type': 'relative', 'column': 'close',
         'operator': '<', 'other_column': 'open', 'recency_n': recency},
        {'id': 'entry', 'name': 'Entry', 'operator': 'or', 'operands': ['up', 'down']},
    ]


def _config(conditions=None, parameters=None):
    return {
        'schema_version': '1.0',
        'metadata': {'name': 'recorded_temporal_parameters', 'limen_version': __version__,
                     'mode': 'development'},
        'sfd': {
            'manifest': {
                'type': 'rule_based',
                'data_source': {'method': 'limen.data.HistoricalData.get_any_file',
                                'params': {'file_path_or_url': str(FIXTURE)}},
                'split_dates': {**dict(zip(DATE_KEYS, DATES, strict=True)),
                                'test_predict_guard': False},
                'strategy': {'conditions': copy.deepcopy(conditions or _conditions()),
                             'entry': 'entry'},
                'reference_architecture': 'limen.sfd.reference_architecture.rule_based.rule_based',
                'backtest': {'fee_bps': '{fee}', 'slip_bps': 2.0, 'notional_rate': '{size}'},
            },
            'params': parameters or {'hold_bars': [1, 3], 'recent_bars': [1, 3],
                                     'fee': [3.0, 10.0], 'size': [0.5, 1.0]},
        },
        'uel': {'n_permutations': 16, 'prep_each_round': True,
                'search_strategy': {'type': 'grid'}},
    }


def _native(conditions):
    return (RuleBasedManifest()
            .set_data_source(HistoricalData.get_any_file,
                             params={'file_path_or_url': str(FIXTURE)})
            .set_split_dates(*[date.fromisoformat(value) for value in DATES],
                             test_predict_guard=False)
            .with_strategy(conditions, entry='entry')
            .with_reference_architecture(rule_based)
            .set_backtest_config(fee_bps='{fee}', slip_bps=2.0, notional_rate='{size}'))


def _rolling_mask(values, window, field):
    result = []
    for index in range(len(values)):
        observed = values[max(0, index - window + 1):index + 1]
        if field == 'persistence_n':
            result.append(len(observed) == window and all(observed))
        else:
            result.append(any(observed))
    return result


def _single_condition(field, value):
    return [{'id': 'entry', 'name': 'Up bar', 'type': 'relative', 'column': 'close',
             'operator': '>', 'other_column': 'open', field: value}]


def _assert_no_artifacts(directory):
    assert not directory.exists() or not list(directory.iterdir())


def test_temporal_templates_match_recorded_literals_and_split_boundaries(recorded_temporal_bars):
    conditions = _conditions()
    config = _config(conditions)
    original = copy.deepcopy(config)
    native = _native(conditions)
    compiled = CompiledSFD(config).manifest()
    assert validate(config).valid
    for window in (1, 3, 1, recorded_temporal_bars.height + 1):
        parameters = {'hold_bars': window, 'recent_bars': window, 'fee': 3.0, 'size': 0.5}
        actual = native.prepare_data(recorded_temporal_bars, parameters)
        yaml_actual = compiled.prepare_data(recorded_temporal_bars, parameters)
        literal = _native(_conditions(window, window)).prepare_data(recorded_temporal_bars, parameters)
        assert actual['_alignment'] == yaml_actual['_alignment'] == literal['_alignment']
        for index, split in enumerate(('train', 'val', 'test')):
            start, end = map(date.fromisoformat, DATES[index * 2:index * 2 + 2])
            rows = recorded_temporal_bars.filter(
                (pl.col('datetime').dt.date() >= start) & (pl.col('datetime').dt.date() < end))
            assert actual[split].equals(yaml_actual[split])
            assert actual[split].equals(literal[split])
            assert actual[split]['close'].equals(rows['close'])
            for field, name, operator in (('persistence_n', 'up', '>'), ('recency_n', 'down', '<')):
                base = rows.select((pl.col('close') > pl.col('open')) if operator == '>'
                                   else (pl.col('close') < pl.col('open'))).to_series().to_list()
                assert actual[split][name].to_list() == _rolling_mask(base, window, field)
            assert not any(actual[split]['up'].head(window - 1))
        assert native.strategy.conditions == conditions
        assert compiled.strategy.conditions == conditions
        assert config == original

    omitted = _conditions(1, 1)
    omitted[0].pop('persistence_n')
    omitted[1].pop('recency_n')
    parameters = {'fee': 3.0, 'size': 0.5}
    baseline = _native(omitted).prepare_data(recorded_temporal_bars, parameters)
    window_one = _native(_conditions(1, 1)).prepare_data(recorded_temporal_bars, parameters)
    for split in ('train', 'val', 'test'):
        assert baseline[split].equals(window_one[split])
    assert baseline['_alignment'] == window_one['_alignment']

    for field, wrapper in (('persistence_n', with_persistence), ('recency_n', with_recency)):
        condition = _single_condition(field, '{window}')[0]
        base = pl.col('close') > pl.col('close').shift(1)
        nullable = {**condition, 'other_column': 'previous_close'}
        frame = recorded_temporal_bars.with_columns(pl.col('close').shift(1).alias('previous_close'))
        expected = frame.select(wrapper(base, 3).alias('result'))
        assert frame.select(build_predicate(nullable, {'window': np.int64(3)}).alias('result')).equals(expected)
        assert frame.select(build_predicate({**nullable, field: np.int64(3)}, {}).alias('result')).equals(expected)
        leaves = [
            {'type': 'threshold', 'column': 'close', 'operator': '>',
             'value': recorded_temporal_bars['open'].median()},
            {'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'},
            {'type': 'crossover', 'column': 'close', 'other_column': 'open'},
            {'type': 'slope', 'column': 'close'},
            {'type': 'sql_expr', 'expr': 'close > open'},
        ]
        for leaf in leaves:
            plain = recorded_temporal_bars.select(wrapper(build_predicate(leaf, {}), 3))
            templated = recorded_temporal_bars.select(build_predicate({**leaf, field: '{window}'}, {'window': 3}))
            assert plain.to_series().equals(templated.to_series())


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('value', (*INVALID, *BAD_REFERENCES))
def test_temporal_modifiers_reject_invalid_declarations(field, value):
    conditions = _single_condition(field, value)
    with pytest.raises(ValueError, match=field):
        RuleBasedConfig(conditions, entry='entry')
    result = validate(_config(conditions))
    assert not result.valid
    assert any(field in str(error) for error in result.errors)
    with pytest.raises(ValueError, match=field):
        build_predicate(conditions[0], {'window': 3})


@pytest.mark.parametrize('field', FIELDS)
def test_temporal_parameter_domains_reject_before_artifacts(field, recorded_temporal_bars, tmp_path):
    wrapper = with_persistence if field == 'persistence_n' else with_recency
    for value in INVALID:
        with pytest.raises(ValueError, match='positive integer'):
            wrapper(pl.col('close') > pl.col('open'), value)
        conditions = _single_condition(field, '{window}')
        parameters = {'window': [1, value], 'fee': [3.0], 'size': [0.5]}
        result = validate(_config(conditions, parameters))
        assert not result.valid, value
        assert any(field in str(error) for error in result.errors)
        manifest = _native(conditions)
        for msq in (False, True):
            directory = tmp_path / f'{field}-{msq}-{type(value).__name__}-{value}'
            sfd = SimpleNamespace(__name__='recorded_temporal_parameters',
                                  manifest=lambda _manifest=manifest: _manifest,
                                  params=lambda _parameters=parameters: _parameters)
            loop = UniversalExperimentLoop(sfd=sfd, data=recorded_temporal_bars,
                experiment_dir=directory,
                search_strategy=GridStrategy(ParamDomain(parameters)) if msq else None)
            with pytest.raises(ValueError, match=field):
                loop.run('invalid', n_permutations=1, prep_each_round=True,
                         random_search=False, progress_bar=False)
            _assert_no_artifacts(directory)
        with pytest.raises(ValueError, match=field):
            build_predicate(conditions[0], {'window': value})

    for parameters in ({}, {'window': [1, 3.0]}):
        result = validate(_config(_single_condition(field, '{window}'),
                                 {'fee': [3.0], 'size': [0.5], **parameters}))
        assert not result.valid
        assert any(field in str(error) for error in result.errors)
    with pytest.raises(ValueError, match=field):
        build_predicate(_single_condition(field, '{window}')[0], {})

    for label, parameters, context in (
        ('missing', {'fee': [3.0], 'size': [0.5]}, None),
        ('context', {'window': [1, 3], 'fee': [3.0], 'size': [0.5]}, {'window': '3'}),
    ):
        manifest = _native(_single_condition(field, '{window}'))
        for msq in (False, True):
            directory = tmp_path / f'{field}-{label}-{msq}'
            sfd = SimpleNamespace(__name__='recorded_temporal_parameters',
                                  manifest=lambda _manifest=manifest: _manifest,
                                  params=lambda _parameters=parameters: _parameters)
            loop = UniversalExperimentLoop(sfd=sfd, data=recorded_temporal_bars,
                experiment_dir=directory,
                search_strategy=GridStrategy(ParamDomain(parameters)) if msq else None)
            with pytest.raises(ValueError, match=field):
                loop.run('invalid', n_permutations=1, prep_each_round=True,
                         context_params=context, random_search=False, progress_bar=False)
            _assert_no_artifacts(directory)

    both = _single_condition(field, 1)
    both[0].update(persistence_n=1, recency_n=1)
    compound = [{'id': 'leaf', 'type': 'relative', 'name': 'Up bar', 'column': 'close',
                 'operator': '>', 'other_column': 'open'},
                {'id': 'entry', 'name': 'Entry', 'operator': 'and',
                 'operands': ['leaf'], field: 3}]
    for conditions in (both, compound):
        with pytest.raises(ValueError, match=r'persistence_n|recency_n'):
            RuleBasedConfig(conditions, entry='entry')
        assert not validate(_config(conditions)).valid

    valid = {'window': [3], 'fee': [3.0], 'size': [0.5]}
    expected = _native(_single_condition(field, 3)).prepare_data(
        recorded_temporal_bars, {'window': 3, 'fee': 3.0, 'size': 0.5})['test']['entry']
    for mode in ('standard-context', 'msq-context', 'standard-params'):
        manifest = _native(_single_condition(field, '{window}'))
        source_parameters = {'fee': [3.0], 'size': [0.5]}
        if mode != 'standard-params':
            source_parameters['window'] = [1, '3']
        sfd = SimpleNamespace(__name__='recorded_temporal_parameters',
                              manifest=lambda _manifest=manifest: _manifest,
                              params=lambda _parameters=source_parameters: _parameters)
        loop = UniversalExperimentLoop(sfd=sfd, data=recorded_temporal_bars,
            experiment_dir=tmp_path / f'{field}-{mode}',
            search_strategy=GridStrategy(ParamDomain(source_parameters)) if mode == 'msq-context' else None)
        overrides = {'params': lambda: valid} if mode == 'standard-params' else {'context_params': {'window': 3}}
        loop.run('effective-window', n_permutations=1, prep_each_round=True,
                 post_processing=True, random_search=False, progress_bar=False, **overrides)
        assert loop.experiment_log['window'].to_list() == [3]
        np.testing.assert_array_equal(loop.preds[0], expected.to_numpy().astype(int))


def _stable(value):
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return 'NaN'
    return value


def test_temporal_factorization_preserves_records_and_signal_reuse(recorded_temporal_bars, tmp_path, monkeypatch):
    config = _config()
    original = copy.deepcopy(config)
    captures = []
    for factorize, expected in ((False, 16), (True, 4)):
        directory = tmp_path / str(factorize)
        counts = {'prep': 0, 'model': 0}
        original_prep, original_model = RuleBasedManifest.prepare_data, RuleBasedManifest.run_model

        def prep(*args, _counts=counts, _original=original_prep, **kwargs):
            _counts['prep'] += 1
            return _original(*args, **kwargs)

        def model(*args, _counts=counts, _original=original_model, **kwargs):
            _counts['model'] += 1
            return _original(*args, **kwargs)

        loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=recorded_temporal_bars,
            search_strategy=build_search_strategy(config), experiment_dir=directory,
            yaml_reference=config)
        with monkeypatch.context() as instrumentation:
            instrumentation.setattr(RuleBasedManifest, 'prepare_data', prep)
            instrumentation.setattr(RuleBasedManifest, 'run_model', model)
            loop.run('recorded_temporal_parameters', n_permutations=16, prep_each_round=True,
                     progress_bar=False, post_processing=True, record_execution=True,
                     factorize=factorize)
        assert counts == {'prep': expected, 'model': expected}
        assert loop.manifest.strategy.conditions == original['sfd']['manifest']['strategy']['conditions']
        with (directory / 'results.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        records = [json.loads(line) for line in _storage.text(_storage.round_path(directory)).splitlines()]
        assert len(rows) == len(records) == 16
        assert len({row['_id'] for row in rows}) == 16
        for row, record in zip(rows, records, strict=True):
            assert record['round_id'] == row['_id']
            hold, recent = int(row['hold_bars']), int(row['recent_bars'])
            test = recorded_temporal_bars.filter(
                (pl.col('datetime').dt.date() >= date(2024, 1, 5)) &
                (pl.col('datetime').dt.date() < date(2024, 1, 7)))
            up = (test['close'] > test['open']).to_list()
            down = (test['close'] < test['open']).to_list()
            expected_predictions = [int(left or right) for left, right in zip(
                _rolling_mask(up, hold, 'persistence_n'),
                _rolling_mask(down, recent, 'recency_n'), strict=True)]
            assert record['preds'] == expected_predictions
            assert record['execution']['pos']
        assert len({tuple(record['preds']) for record in records}) > 1
        assert len({row['pnl_per_bar_bps_test'] for row in rows}) > 1
        captures.append(_stable({
            'rows': [{key: value for key, value in row.items() if key != 'execution_time'}
                     for row in rows], 'records': records,
        }))
        assert config == original
    assert captures[0] == captures[1]

    overlapping = _config(_conditions('{fee}', '{recent_bars}'))
    overlapping['sfd']['params']['fee'] = [1, 3]
    directory = tmp_path / 'economic-overlap'
    loop = UniversalExperimentLoop(sfd=CompiledSFD(overlapping), data=recorded_temporal_bars,
        search_strategy=build_search_strategy(overlapping), experiment_dir=directory,
        yaml_reference=overlapping)
    with pytest.raises(ValueError, match=r'backtest parameter fee.*manifest'):
        loop.run('economic-overlap', n_permutations=1, prep_each_round=True,
                 progress_bar=False, factorize=True)
    _assert_no_artifacts(directory)
