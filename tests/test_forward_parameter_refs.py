import copy
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from threadpoolctl import threadpool_limits

from docs.examples.tests.helper_oracles import verify_rules
from limen.cli.main import cli
from limen.experiment import UniversalExperimentLoop
from limen.experiment.manifest_core import _resolve_params
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from limen.yaml.config import round_trip_yaml
from limen.yaml.rules import ParamCoverage

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


def _config():
    config, errors = parse(ROOT / 'docs/examples/compiled-research/target_nextbarup.yaml')
    assert not errors
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    config['sfd']['params'].pop('C')
    return config


@pytest.fixture(scope='module')
def bars():
    return pl.read_parquet(FIXTURE)


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def architecture(monkeypatch):
    module = importlib.import_module('limen.sfd.reference_architecture.logreg_binary')
    observed = []

    def adapter(data, **params):
        observed.append(dict(params))
        supported = inspect.signature(module.logreg_binary).parameters
        return module.logreg_binary(data, **{key: value for key, value in params.items() if key in supported})

    adapter.__module__ = module.__name__
    monkeypatch.setattr(module, 'reference_probe', adapter, raising=False)
    return adapter, observed


def _signature(config, kind, architecture, monkeypatch):
    if kind == 'ordinary':
        return
    adapter, _ = architecture
    config['sfd']['manifest']['reference_architecture'] = adapter.__module__ + '.reference_probe'
    if kind == 'unknown':
        original = inspect.signature

        def unknown(obj, *args, **kwargs):
            if obj is adapter:
                raise ValueError('Signature unavailable for this callable')
            return original(obj, *args, **kwargs)

        monkeypatch.setattr(inspect, 'signature', unknown)


@pytest.mark.parametrize('name', ['C', 'feature_groups', 'auto_pca', 'pca_k', 'feature_drop_count',
                                  'custom_drop', 'use_calibration', 'use_threshold', 'scaler_type'])
def test_forward_lookup_requires_a_real_declaration(name):
    config = _config()
    config['sfd']['params'].pop('scaler_type')
    config['sfd']['manifest'].update(pca_compression={},
        feature_ablation={'drop_count_key': 'custom_drop'} if name == 'custom_drop' else {},
        calibration={'probability_calibration': {'func': 'limen.calibration.sklearn_probability_calibrator'}})
    config['sfd']['manifest']['indicators'][0]['params']['period'] = '{' + name + '}'
    result = validate(config)
    assert not result.valid
    assert any(error.path == 'sfd.manifest.indicators[0].params.period' and
        f"'{name}' is not defined in sfd.params" in error.message for error in result.errors)


@pytest.mark.parametrize('kind', ['ordinary', 'kwargs', 'unknown'])
def test_signature_uncertainty_never_suppresses_forward_checks(kind, architecture, monkeypatch):
    config = _config()
    _signature(config, kind, architecture, monkeypatch)
    config['sfd']['manifest']['indicators'][0]['params']['period'] = '{C}'
    result = validate(config)
    error = next(error for error in result.errors if "'C' is not defined" in error.message)
    assert error.path == 'sfd.manifest.indicators[0].params.period'
    assert ('also overrides architecture' in error.suggestion) is (kind == 'ordinary')
    config['sfd']['params']['C'] = [3]
    assert validate(config).valid


@pytest.mark.parametrize('kind', ['ordinary', 'kwargs', 'unknown'])
@pytest.mark.parametrize('values', [None, [0, 1], [False, True]])
def test_include_if_remains_checked_for_all_signatures(kind, values, architecture, monkeypatch):
    config = _config()
    _signature(config, kind, architecture, monkeypatch)
    config['sfd']['manifest']['indicators'][0]['include_if'] = 'enable_roc'
    if values is not None:
        config['sfd']['params']['enable_roc'] = values
    result = validate(config)
    assert result.valid is (values is not None and all(type(value) is bool for value in values))
    if values is None:
        assert any("'enable_roc' is not defined" in error.message for error in result.errors)
    elif type(values[0]) is int:
        assert any('only boolean' in error.message for error in result.errors)


def test_include_if_without_any_parameter_or_brace_reference_is_not_skipped():
    config = _config()
    config['sfd']['params'] = {}
    config['sfd']['manifest']['indicators'][0]['include_if'] = 'enabled'
    assert any("'enabled' is not defined" in error.message for error in validate(config).errors)


@pytest.mark.parametrize('kind', ['ordinary', 'kwargs', 'unknown'])
def test_reverse_warnings_require_a_known_closed_signature(kind, architecture, monkeypatch):
    config = _config()
    _signature(config, kind, architecture, monkeypatch)
    config['sfd']['params']['unrelated_setting'] = [3]
    result = validate(config)
    assert result.valid
    assert any(warning.path == 'sfd.params.unrelated_setting' for warning in result.warnings) is (kind == 'ordinary')


@pytest.mark.parametrize('kind', ['ordinary', 'kwargs'])
def test_declared_lookup_prepares_recorded_roc_and_trains_real_model(bars, kind, architecture, monkeypatch):
    config = _config()
    _signature(config, kind, architecture, monkeypatch)
    config['sfd']['params']['C'] = [3]
    config['sfd']['manifest']['indicators'][0]['params']['period'] = '{C}'
    config['sfd']['manifest'].pop('scaler')
    assert validate(config).valid
    compiled = CompiledSFD(config)
    manifest = compiled.manifest()
    params = {key: values[0] for key, values in compiled.params().items()}
    prepared = manifest.prepare_data(bars, params)
    expected = bars.with_columns(((pl.col('close') / pl.col('close').shift(3) - 1) * 100).alias('roc_3'))
    aligned = prepared['price_data_for_backtest'].select('datetime').join(expected, on='datetime')
    np.testing.assert_allclose(prepared['x_test']['roc_3'].to_numpy(), aligned['roc_3'].to_numpy(), rtol=1e-12, atol=1e-12)
    result = manifest.run_model(prepared, params)
    assert result['_model'].model.C == 3
    assert len(result['_preds']) == 743
    if kind == 'kwargs':
        assert architecture[1][-1]['C'] == 3


@pytest.mark.parametrize('literal', ['group', 'source', 'override'])
def test_literal_braces_preserve_real_execution_and_model_default(bars, tmp_path, literal):
    config = _config()
    manifest_config = config['sfd']['manifest']
    if literal == 'group':
        manifest_config['indicators'][0]['params']['group'] = 'momentum_{C}'
    elif literal == 'source':
        path = tmp_path / 'recorded_{C}.parquet'
        path.write_bytes(FIXTURE.read_bytes())
        manifest_config['data_source']['params']['file_path_or_url'] = str(path)
    else:
        manifest_config['data_source']['params']['file_path_or_url'] = str(tmp_path / 'unused_{C}.parquet')
        manifest_config['params_override'] = {'file_path_or_url': str(FIXTURE)}
    assert validate(config).valid
    manifest = CompiledSFD(config).manifest()
    if literal != 'group':
        assert manifest.fetch_data().equals(bars)
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    prepared = manifest.prepare_data(bars, params)
    result = manifest.run_model(prepared, params)
    assert result['_model'].model.C == 1.0
    assert len(result['_preds']) == 743
    manifest_config['indicators'][0]['params']['period'] = '{C}'
    assert not validate(config).valid  # A literal occurrence cannot hide a real lookup.


def test_native_literal_and_exact_key_resolution_precedence():
    config = _config()
    params = config['sfd']['manifest']['indicators'][0]['params']
    params['period'] = '_literal_{C}'
    assert validate(config).valid
    assert _resolve_params({'period': params['period']}, {}) == {'period': '_literal_{C}'}
    params['period'] = 'period_{C}'
    config['sfd']['params']['period_{C}'] = [3]
    assert validate(config).valid
    assert _resolve_params({'period': params['period']}, {'period_{C}': 3}) == {'period': 3}


def test_target_embedded_reference_uses_the_declared_recorded_column(bars):
    config = _config()
    config['sfd']['manifest']['indicators'][0]['params']['period'] = 3
    config['sfd']['manifest']['target'] = {'name': 'label', 'class': 'limen.targets.ThresholdBinaryTarget',
        'fit_params': {'source_column': 'roc_{C}', 'threshold': 0}, 'transform_params': {'shift': 0}}
    result = validate(config)
    assert not result.valid
    assert any(error.path == 'sfd.manifest.target.fit_params.source_column' and
               "'C' is not defined" in error.message for error in result.errors)
    config['sfd']['params']['C'] = [3]
    assert validate(config).valid
    compiled = CompiledSFD(config)
    manifest = compiled.manifest()
    prepared = manifest.prepare_data(bars, {key: values[0] for key, values in compiled.params().items()})
    expected = bars.with_columns((pl.col('close') > pl.col('close').shift(3)).cast(pl.UInt8).alias('expected'))
    aligned = prepared['price_data_for_backtest'].select('datetime').join(expected, on='datetime')
    np.testing.assert_array_equal(prepared['y_test'].to_numpy().ravel(), aligned['expected'].to_numpy())


def test_rule_architecture_default_is_not_a_predicate_declaration(bars):
    config, errors = parse(ROOT / 'docs/examples/compiled-research/rule_contracts.yaml')
    assert not errors
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    config['sfd']['manifest']['strategy']['conditions'][0]['value'] = '{sharpe_std_threshold}'
    config['sfd']['params'] = {}
    result = validate(config)
    assert not result.valid
    assert any(error.path == 'sfd.manifest.strategy.conditions[0].value' and
               "'sharpe_std_threshold' is not defined" in error.message for error in result.errors)
    config['sfd']['params']['sharpe_std_threshold'] = [40000., 41000.]
    assert validate(config).valid
    manifest = CompiledSFD(config).manifest()
    rows = []
    for floor in config['sfd']['params']['sharpe_std_threshold']:
        params = {'sharpe_std_threshold': floor}
        prepared = manifest.prepare_data(bars, params)
        signals = manifest.run_model(prepared, params)['_preds']
        assert len(signals) == 96 and sum(signals) == 32
        rows.append({'price_floor': floor, 'signals': signals, 'positive_decisions': sum(signals)})
    verify_rules({'assessment_rows': 96, 'history_resets_at_split': True, 'rows': rows}, FIXTURE.parent)


@pytest.mark.parametrize('inactive', ['include_if', 'groups'])
def test_inactive_transform_still_requires_declaration(inactive):
    config = _config()
    config['sfd']['manifest']['indicators'][0]['params']['period'] = '{C}'
    if inactive == 'include_if':
        config['sfd']['manifest']['indicators'][0]['include_if'] = 'enabled'
        config['sfd']['params']['enabled'] = [False]
    else:
        config['sfd']['params']['feature_groups'] = [['another_group']]
    assert any("'C' is not defined" in error.message for error in validate(config).errors)


def test_disabled_calibration_still_requires_declared_lookup():
    config = _config()
    config['sfd']['manifest']['calibration'] = {'probability_calibration': {
        'func': 'limen.calibration.sklearn_probability_calibrator', 'params': {'cv': '{C}'}}}
    config['sfd']['params']['use_calibration'] = [False]
    assert any(error.path == 'sfd.manifest.calibration.probability_calibration.params.cv' and
               "'C' is not defined" in error.message for error in validate(config).errors)


@pytest.mark.parametrize('command', ['validate', 'dry-run', 'run'])
def test_cli_missing_reference_precedes_compilation_and_writes(tmp_path, monkeypatch, command):
    config = _config()
    config['sfd']['manifest']['indicators'][0]['params']['period'] = '{C}'
    config['uel']['output_path'] = str(tmp_path / 'forbidden-output')
    path = tmp_path / 'config.yaml'
    with path.open('w') as stream:
        round_trip_yaml().dump(config, stream)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid declaration reached compilation')

    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    args = ['run', '--dry-run', str(path)] if command == 'dry-run' else [command, str(path)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert "'C' is not defined in sfd.params" in result.output
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('declared', [False, True])
@pytest.mark.parametrize('native', [False, True])
def test_context_overrides_still_work_without_changing_declared_domain(bars, tmp_path, monkeypatch, declared, native):
    config = _config()
    config['sfd']['manifest']['indicators'][0]['params']['period'] = '{C}'
    if declared:
        config['sfd']['params']['C'] = [1]
        assert validate(config).valid
    compiled = CompiledSFD(config)  # Direct Python callers may supply the missing key through context.
    manifest, domain = compiled.manifest(), compiled.params()
    original = manifest.run_model
    observed = []

    def run_model(data, params):
        assert 'roc_3' in data['x_test'].columns
        result = original(data, params)
        observed.append(result['_model'].model.C)
        return result

    monkeypatch.setattr(manifest, 'run_model', run_model)
    sfd = SimpleNamespace(__name__='reference_context', manifest=lambda: manifest, params=lambda: copy.deepcopy(domain)) if native else compiled
    loop = UniversalExperimentLoop(sfd=sfd, data=bars, experiment_dir=tmp_path,
                                    search_strategy=build_search_strategy(config))
    loop.run('reference_context', n_permutations=1, prep_each_round=True, progress_bar=False,
             context_params={'C': 3}, record_model_outputs=True)
    assert loop.experiment_log['C'].to_list() == [3]
    assert observed == [3]
    assert compiled.params() == domain
    assert domain.get('C') == ([1] if declared else None)


@pytest.mark.parametrize('manifest, path', [
    ({'type': 'ml', 'features': [{'params': {'window': '{C}'}}]}, 'features[0].params.window'),
    ({'type': 'ml', 'target': {'fit_params': {'source_column': 'roc_{C}'}}}, 'target.fit_params.source_column'),
    ({'type': 'ml', 'target': {'transform_params': {'shift': '{C}'}}}, 'target.transform_params.shift'),
    ({'type': 'ml', 'scaler': {'params': {'window': '{C}'}}}, 'scaler.params.window'),
    ({'type': 'ml', 'pre_split_data_selector': {'params': {'n_rows': '{C}'}}}, 'pre_split_data_selector.params.n_rows'),
    ({'type': 'ml', 'bar_formation': {'params': {'n_bars': '{C}'}}}, 'bar_formation.params.n_bars'),
    ({'type': 'ml', 'calibration': {'threshold_function': {'params': {'step': '{C}'}}}}, 'calibration.threshold_function.params.step'),
    ({'type': 'ml', 'calibration': {'probability_calibration': {'params': {'cv': '{C}'}}}}, 'calibration.probability_calibration.params.cv'),
    ({'type': 'ml', 'feature_shims': [{'columns': ['roc_{C}'], 'method': 'log1p'}]}, 'feature_shims[0].columns[0]'),
    ({'type': 'rule_based', 'strategy': {'conditions': [{'value': '{C}'}]}}, 'strategy.conditions[0].value'),
    ({'backtest': {'initial_equity': '{C}'}}, 'backtest.initial_equity'),
    ({'backtest': {'product': {'quantity_step': '{C}'}}}, 'backtest.product.quantity_step'),
    ({'allocation': {'params': {'nested': [{'size': '{C}'}]}}}, 'allocation.params.nested[0].size'),
    ({'backtest': {'funding': {'params': {'rate': '{C}'}}}}, 'backtest.funding.params.rate'),
    ({'backtest': {'barrier_features': {'params': {'window': '{C}'}}}}, 'backtest.barrier_features.params.window'),
])
def test_lookup_locations_report_the_actual_consumer(manifest, path):
    errors, warnings = [], []
    ParamCoverage().check({'sfd': {'manifest': manifest, 'params': {}}}, errors, warnings)
    assert len(errors) == 1
    assert errors[0].path == 'sfd.manifest.' + path
    assert "'C' is not defined" in errors[0].message
