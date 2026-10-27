from copy import deepcopy
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from limen.experiment._allocation import AllocationBatch
from limen.experiment._allocation_folds import fit_primary, training_opportunities
from limen.experiment.errors import StrictModeError
from limen.sfd.reference_architecture.meta_allocation import MetaAllocationModel
from limen.yaml import CompiledSFD, validate
from tests.test_dynamic_barriers import FIXTURE, _yaml


def recorded_primary(data, **params):
    from inspect import signature
    from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary
    assert not any(key.startswith(('_trade', '_allocation')) for key in data)
    assert not data['_alignment']
    assert 'price_data_for_backtest' not in data
    assert data['x_test'].equals(data['x_val'])
    assert data['y_test'].equals(data['y_val'])
    assert params['risk_span'] == 6
    return lightgbm_binary(data, **{key: value for key, value in params.items() if key in signature(lightgbm_binary).parameters})


def seeded_primary(data, **params):
    assert params['seed'] == 73
    params['random_state'] = params['seed']
    return recorded_primary(data, **params)


def _config(ml=False):
    config = _yaml()
    declared = config['sfd']['manifest']
    declared['split_dates'] = {'train_start': '2024-01-02', 'train_end': '2024-01-15', 'val_start': '2024-01-15', 'val_end': '2024-01-17', 'test_start': '2024-01-17', 'test_end': '2024-01-19'}
    declared['backtest'].update(prediction_mode='target_exposure', fee_bps=0, slip_bps=0, max_holding_seconds=7200)
    declared['allocation'] = {'method': 'limen.sfd.reference_architecture.meta_allocation.meta_allocation', 'params': {'folds': 3, 'min_train_samples': 2, 'features': ['open', 'close'], 'probability_threshold': 0.1}}
    if ml:
        declared['type'] = 'ml'
        declared.pop('strategy')
        declared['indicators'] = [{'func': 'limen.indicators.window_return', 'params': {'period': 1}}]
        declared['target'] = {'name': 'forward', 'class': 'limen.targets.QuantileBinaryTarget', 'fit_params': {'source_column': 'ret_1', 'quantile': 0.5}, 'transform_params': {'shift': -1}}
        declared['reference_architecture'] = 'limen.sfd.reference_architecture.lightgbm_binary.lightgbm_binary'
        config['sfd']['params'].update(n_estimators=[5], min_child_samples=[3], num_leaves=[3], early_stopping_rounds=[2], n_jobs=[1])
        if ml == 'logreg':
            declared['reference_architecture'] = 'limen.sfd.reference_architecture.logreg_binary.logreg_binary'
            declared['scaler'] = {'class': 'limen.scalers.RobustScaler'}
    return config


def _prepared(ml=False):
    config = _config(ml)
    manifest = CompiledSFD(config).manifest()
    raw = pl.read_parquet(FIXTURE, n_rows=480)
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    return manifest, raw, params, manifest.prepare_data(raw, params)


@pytest.mark.parametrize('ml', [False, True, 'logreg'])
def test_rule_and_ml_primary_identity_is_preserved(ml):
    manifest, raw, params, data = _prepared(ml)
    original = deepcopy(manifest)
    original.allocation_config = None
    original.backtest_config.prediction_mode = 'binary'
    expected = original.prepare_data(raw, params)
    result = manifest.run_model(data, params)
    model = result['_model']
    assert isinstance(model, MetaAllocationModel)
    plain = fit_primary(original, expected, params)
    batch = data['_allocation_batches']['test']
    np.testing.assert_array_equal(model.primary.predict(dict(batch.primary_inputs))['_preds'], plain.predict(dict(batch.primary_inputs))['_preds'])
    exposure = result['_preds']
    primary = model.primary.predict(dict(batch.primary_inputs))['_preds']
    assert np.all(exposure[primary == 0] == 0)
    assert result['allocation_training_completed'] >= 2
    repeated = model.predict({'allocation_batch': batch})
    np.testing.assert_array_equal(exposure, repeated['_preds'])


def test_prefix_refits_and_completed_outcome_purging():
    manifest, _, _, data = _prepared(True)
    context = data['_allocation_context']
    labels = training_opportunities(context, folds=3, validation_fraction=0.2, include_primary_proba=True)
    completed = labels.rows.filter(pl.col('available'))
    assert completed.height >= 2
    for block in labels.binding['folds']:
        opened = completed.filter(pl.col('opening_block') == block['block'])
        assert opened['label_available_ns'].lt(block['end_ns']).all()
        assert block['fit_prefix_rows'] < context.raw_train.height
    from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
    indices = np.array_split(np.arange(context.raw_train.height), 3)[1]
    start, end = int(indices[0]), int(indices[-1]) + 1
    prefix = context.raw_train.head(start - 1)
    original = deepcopy(context.manifest)
    original.backtest_config = None
    original.split_dates = None
    original.split_config = (int(prefix.height * 0.8), int(prefix.height * 0.2), 0)
    independent = original.prepare_data(prefix, dict(context.round_params))
    fitted = LightGBMBinary().train(independent, **original.resolve_model_kwargs(dict(context.round_params)))
    inference, _ = original.sensor_input_prep(context.raw_train.head(end), independent['_fitted_params'], dict(context.round_params))
    inference = inference.filter(pl.col('datetime').is_between(context.raw_train['datetime'][start], context.raw_train['datetime'][end - 1])).select(independent['x_train'].columns).drop_nulls()
    predictions = fitted.predict({'x_test': inference})['_preds']
    from limen.backtest.trade_contract import contract_digest
    assert labels.binding['folds'][0]['primary_digest'] == contract_digest({'predictions': predictions.astype(float).tolist()})
    assert labels.binding['folds'][0]['member_count'] == len(indices)
    assert labels.rows['opportunity_id'].n_unique() == labels.rows.height
    np.testing.assert_allclose(completed['net_return_on_entry_notional'], completed['net_pnl'] / completed['entry_notional'])
    assert completed['net_return_on_entry_notional'].gt(0).any()
    assert completed['net_return_on_entry_notional'].lt(0).any()
    assert 'primary_probability' in labels.features.columns
    assert manifest.target_class_config.fit_params == context.manifest.target_class_config.fit_params


def test_once_per_opportunity_allocation_and_risk_caps():
    manifest, _, params, data = _prepared()
    model = manifest.run_model(data, params)['_model']
    model.context = replace(model.context, policy=replace(model.context.policy, fee_bps=10, slip_bps=5, max_exposure=0.8, flat_threshold=0.001))
    model.risk, model.max_size, model.return_scale, model.threshold = 0.002, 0.9, 0.01, 0.5
    assert model._allocation(0.5, 0.05, 70) == pytest.approx(0.2)
    assert model._allocation(0.49, 0.05, 70) == 0
    assert model._allocation(1, -0.01, 70) == 0
    with pytest.raises(ValueError, match='positive decision-time stop'):
        model._allocation(1, 1, None)
    batch = data['_allocation_batches']['test']
    output = model.predict({'allocation_batch': batch})
    primary, exposures = output['primary']['_preds'], output['_preds']
    for index in range(1, len(primary)):
        if primary[index] == primary[index - 1] == 1:
            assert exposures[index].hex() == exposures[index - 1].hex()
    model.context = replace(model.context, policy=replace(model.context.policy, fee_bps=10, slip_bps=15))
    assert model._allocation(1, 1, 70) == pytest.approx(0.002 / 0.012)


def test_yaml_cli_allocation_contract(tmp_path):
    from click.testing import CliRunner
    from ruamel.yaml import YAML
    from limen.cli.main import cli

    config = _config()
    assert validate(config).valid
    for malformed in (123, [], 'invalid', {}):
        invalid = deepcopy(config)
        invalid['sfd']['manifest']['allocation'] = malformed
        assert not validate(invalid).valid
    path = tmp_path / 'allocation.yaml'
    with path.open('w') as stream:
        YAML().dump(config, stream)
    for command in (['validate', str(path)], ['run', str(path), '--dry-run']):
        result = CliRunner().invoke(cli, command)
        assert result.exit_code == 0, result.output
    manifest, raw, params, _ = _prepared()
    from types import SimpleNamespace
    from limen.experiment._allocation_folds import admit
    manifest._holdout_binding = SimpleNamespace(cutoff_ns=None)
    admit(manifest, params)
    manifest._holdout_binding = None
    with pytest.raises(ValueError, match='already configured'):
        manifest.with_allocation(manifest.allocation_config.method)
    for key, value in (('execution_lag_seconds', 1), ('notional_rate', 0.5)):
        other = deepcopy(manifest)
        setattr(other.backtest_config, key, value)
        with pytest.raises(ValueError, match='zero lag'):
            other.prepare_data(raw, params)
    with pytest.raises(StrictModeError, match='Insufficient completed'):
        manifest.allocation_config.method(manifest.prepare_data(raw, params), context=manifest.prepare_data(raw, params)['_allocation_context'], min_train_samples=100000)


def test_research_promotion_rejects_before_training(tmp_path):
    import json
    from unittest.mock import patch
    from limen.inference import Trainer, Sensor
    from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy

    config = _config()
    backtest = config['sfd']['manifest']['backtest']
    for key in ('take_profit_bps_column', 'stop_loss_bps_column', 'barrier_features'):
        backtest.pop(key)
    backtest['stop_loss_bps'] = 100
    (tmp_path / 'metadata.json').write_text(json.dumps({'yaml_reference': config}))
    with patch.object(Trainer, '_load_round_data', side_effect=AssertionError('late rejection')), pytest.raises(ValueError, match='research-only'):
        Trainer(tmp_path)
    with pytest.raises(ValueError, match='research-only'):
        Sensor(config, RuleBasedStrategy(), {}, {})


def test_recording_omissions_and_membership_digests():
    from limen.experiment._recording import project_record

    manifest, _, params, data = _prepared()
    data['_record_model_outputs'] = True
    result = manifest.run_model(data, params)
    model = result['_model']
    batch = data['_allocation_batches']['test']
    assert isinstance(batch, AllocationBatch)
    for invalid in (replace(batch, row_ids=batch.row_ids[::-1]), replace(batch, meta_features=batch.meta_features.reverse()), replace(batch, stop_loss_bps=tuple(1 for _ in batch.row_ids))):
        with pytest.raises(ValueError, match='complete original bound'):
            model.predict({'allocation_batch': invalid})
    with pytest.raises(ValueError, match='complete original bound'):
        model.predict({'x_test': batch.meta_features})
    assert data['_alignment']['model_outputs']['architecture'] == 'meta_allocation'
    assert 'allocation_training' in data['_alignment']['trade_ledger']
    recorded = {'trade_ledger': data['_alignment']['trade_ledger'], 'model_outputs': data['_alignment']['model_outputs']}
    omitted = project_record(recorded, {'predictions': True, 'alignment': True, 'trade_ledger': False, 'fold_details': True})
    assert 'trade_ledger' not in omitted
    assert model.learning_binding['mapping_version'] == 'primary_opportunity_signed_net_notional_v1'


def test_crash_reversal_declared_clock_preserves_nonterminal_prefixes():
    from tests.test_dollar_bar_crash_reversal import _candidate
    from limen.features import dollar_bar_crash_reversal
    raw = pl.read_parquet(FIXTURE)
    configured = dollar_bar_crash_reversal(raw, -575, -0.5, 60, bar_interval_seconds=3600)
    column = 'dollar_bar_crash_reversal_position'
    assert configured[column].head(raw.height - 1).equals(_candidate(raw)[column].head(raw.height - 1))
    assert configured[column].sum() > 0
    for end in (raw.height // 3, raw.height // 2, raw.height - 1):
        prefix = dollar_bar_crash_reversal(raw.head(end), -575, -0.5, 60, bar_interval_seconds=3600)
        assert prefix[column].equals(configured[column].head(end))
    with pytest.raises(ValueError, match='verified regular'):
        dollar_bar_crash_reversal(raw, -575, -0.5, 60, bar_interval_seconds=3601)


def test_primary_fitting_excludes_later_batch_evidence():
    from unittest.mock import patch
    manifest, raw, params, _ = _prepared(True)
    manifest.with_reference_architecture(recorded_primary)
    data = manifest.prepare_data(raw, params)
    model = manifest.run_model(data, params)['_model']
    batch = data['_allocation_batches']['test']
    expected = model.predict({'allocation_batch': batch})['_preds']
    with patch.object(model.primary, 'predict', side_effect=AssertionError('Bound batch must not reevaluate a mutable primary')):
        np.testing.assert_array_equal(model.predict({'allocation_batch': batch})['_preds'], expected)


def test_component_probabilities_reject_invalid_nonentry_values():
    from unittest.mock import patch
    manifest, _, params, data = _prepared()
    model = manifest.run_model(data, params)['_model']
    batch = data['_allocation_batches']['test']
    primary = model.primary.predict(dict(batch.primary_inputs))['_preds']
    flat = np.flatnonzero(primary == 0)[0]
    probabilities = np.full((len(primary), 2), 0.5)
    probabilities[flat] = [-0.1, 1.1]
    with patch.object(model.probability_model, 'predict_proba', return_value=probabilities), pytest.raises(ValueError, match='Malformed take probabilities'):
        model._compose(batch)


@pytest.mark.parametrize('ml', [False, True])
def test_recorded_availability_columns_never_become_meta_features(ml):
    from limen.experiment._prepare_trade_context import normalize_observations
    config = _config(ml)
    config['sfd']['manifest']['allocation']['params']['features'] = None
    manifest = CompiledSFD(config).manifest()
    raw = pl.read_parquet(FIXTURE, n_rows=480)
    timing = normalize_observations(raw, interval_seconds=3600)
    columns = ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns')
    raw = raw.with_columns(timing.select(columns))
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    data = manifest.prepare_data(raw, params)
    model = manifest.run_model(data, params)['_model']
    assert not set(columns) & set(model.feature_names)
    assert len(model.predict({'allocation_batch': data['_allocation_batches']['test']})['_preds']) == data['_trade_inputs'].signals.height


@pytest.mark.parametrize('folds', [2, 3])
@pytest.mark.parametrize('ml', [False, True])
def test_oof_rejects_terminal_lookahead(folds, ml):
    manifest, raw, params, _ = _prepared(ml)
    def terminal_lookahead(frame):
        return frame.with_columns((pl.col('close') / pl.col('close').shift(-1) - 1.).alias('ret_-1'))
    manifest.add_indicator(terminal_lookahead)
    data = manifest.prepare_data(raw, params)
    with pytest.raises(ValueError, match='observations after their decision'):
        training_opportunities(data['_allocation_context'], folds=folds, validation_fraction=0.2, include_primary_proba=False)


def test_opportunity_exit_keeps_next_open_at_the_flat_clock():
    from limen.experiment._allocation_folds import _opportunity
    from limen.experiment._prepare_trade_context import normalize_observations
    _, raw, _, data = _prepared()
    prices = normalize_observations(raw.head(8), interval_seconds=3600)
    assert prices['close'][1] != prices['open'][2]
    context = data['_allocation_context']
    policy = replace(context.policy, take_profit_bps_column=None, stop_loss_bps_column=None, barrier_spec=None, max_holding_seconds=None)
    context = replace(context, policy=policy, barriers=None)
    row, available = _opportunity(context, prices, raw.head(2), np.array([1.0, 0.0]), 0, 1, int(prices['available_at_ns'][-1]))
    assert available
    assert row['closed_at_ns'] == prices['start_ns'][2]
    assert row['net_return_on_entry_notional'] == pytest.approx(prices['open'][2] / prices['open'][1] - 1)


def test_opportunity_funding_is_charged_and_source_bound():
    from limen.experiment._resolve_trade_policy import FundingConfig
    manifest, raw, params, _ = _prepared()
    manifest.backtest_config.product.kind = 'linear_perpetual'
    manifest.backtest_config.funding = FundingConfig(preset='binance_btcusdt', params={'rate': 0.001})
    context = manifest.prepare_data(raw, params)['_allocation_context']
    funded = training_opportunities(context, folds=3, validation_fraction=0.2, include_primary_proba=False)
    plain = replace(context, policy=replace(context.policy, funding=None), train_inputs=replace(context.train_inputs, funding_events=None))
    unfunded = training_opportunities(plain, folds=3, validation_fraction=0.2, include_primary_proba=False)
    rows = funded.rows.filter(pl.col('available'))
    others = unfunded.rows.filter(pl.col('available'))
    assert rows['opportunity_id'].equals(others['opportunity_id'])
    assert rows['net_pnl'].le(others['net_pnl']).all()
    assert rows['net_pnl'].lt(others['net_pnl']).any()


def test_explicit_empty_component_params_select_factory_defaults():
    from unittest.mock import patch
    from sklearn.linear_model import LogisticRegression, Ridge
    manifest, _, params, data = _prepared()
    allocation = manifest.allocation_config
    allocation.params.update(probability_params={}, sizing_params={})
    with patch('limen.sfd.reference_architecture.direction_sizing.LogisticRegression', wraps=LogisticRegression) as probability, patch('limen.sfd.reference_architecture.direction_sizing.Ridge', wraps=Ridge) as sizing:
        manifest.run_model(data, params)
    assert probability.call_args.kwargs == {'random_state': 42}
    assert sizing.call_args.kwargs == {'random_state': 42}


def test_allocation_reference_preserves_shared_primary_kwargs():
    manifest, raw, params, _ = _prepared(True)
    manifest.with_reference_architecture(seeded_primary)
    manifest.allocation_config.params['seed'] = '{seed}'
    params['seed'] = 73
    original = deepcopy(manifest)
    original.allocation_config = None
    assert original.resolve_model_kwargs(params) == manifest.resolve_model_kwargs(params)
    data = manifest.prepare_data(raw, params)
    model = manifest.run_model(data, params)['_model']
    assert model.primary.model.get_params()['random_state'] == 73
    assert model.learning_binding['primary_params']['seed'] == 73


@pytest.mark.parametrize('ml', [False, True])
def test_end_stamped_gapped_source_preserves_prices_and_decision_clocks(ml):
    from limen.experiment._prepare_trade_context import normalize_observations, sensor_decisions
    config = _config(ml)
    config['sfd']['manifest']['backtest']['source_timestamp_role'] = 'end'
    assert validate(config).valid
    manifest = CompiledSFD(config).manifest()
    original = pl.read_parquet(FIXTURE, n_rows=480).with_row_index().filter(~pl.col('index').is_in([30, 31, 160])).drop('index')
    encoded = original.with_columns(pl.col('datetime') + pl.duration(hours=1))
    expected = normalize_observations(original.with_columns(
        pl.col('datetime').dt.epoch('ns').alias('start_ns'),
        (pl.col('datetime').dt.epoch('ns') + 3600000000000).alias('end_ns'),
        pl.col('datetime').dt.epoch('ns').alias('open_available_at_ns'),
        (pl.col('datetime').dt.epoch('ns') + 3600000000000).alias('available_at_ns')))
    actual = normalize_observations(encoded, interval_seconds=3600, timestamp_role='end')
    assert actual.equals(expected)
    assert actual.height == original.height
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    data = manifest.prepare_data(encoded, params)
    for rows in data['_trade_context'].model_rows:
        assert rows['available_at_ns'].equals(rows['datetime'].dt.epoch('ns').rename('available_at_ns'))
    context = data['_allocation_context']
    labels = training_opportunities(context, folds=3, validation_fraction=0.2, include_primary_proba=False)
    assert labels.rows['first_fill_ns'].ge(labels.rows['decision_available_at_ns']).drop_nulls().all()
    model = manifest.run_model(data, params)['_model']
    assert model.context.policy.source_timestamp_role == 'end'
    metadata = sensor_decisions(encoded, encoded, interval_seconds=3600, timestamp_role='end')
    assert metadata['__trade_available_at_ns__'].to_list() == encoded['datetime'].dt.epoch('ns').to_list()
    from types import SimpleNamespace
    from limen.experiment._fold_partitions import decisions
    folded = decisions(SimpleNamespace(_fold_native_interval=3600000000000), encoded, encoded, interval_seconds=3600, native=sensor_decisions, timestamp_role='end')
    assert folded.equals(metadata)


def test_hour_end_configuration_rejects_missing_or_incoherent_metadata():
    from limen.experiment._prepare_trade_context import normalize_observations
    raw = pl.read_parquet(FIXTURE, n_rows=12)
    for invalid in ('close', None, 1):
        config = _config()
        config['sfd']['manifest']['backtest']['source_timestamp_role'] = invalid
        assert not validate(config).valid
    config = _config()
    config['sfd']['manifest']['backtest'].update(source_timestamp_role='end', source_interval_seconds=None)
    assert not validate(config).valid
    with pytest.raises(ValueError, match='verified regular'):
        normalize_observations(raw, interval_seconds=3601, timestamp_role='end')


def test_declared_crash_clock_preserves_gaps_and_prefixes():
    from limen.features import dollar_bar_crash_reversal
    raw = pl.read_parquet(FIXTURE).with_row_index().filter(~pl.col('index').is_in([30, 31, 160])).drop('index')
    actual = dollar_bar_crash_reversal(raw, -575, -0.5, 60, bar_interval_seconds=3600)
    assert actual.height == raw.height
    assert actual['dollar_bar_crash_reversal_position'].sum() > 0
    for end in (raw.height // 3, raw.height // 2):
        prefix = dollar_bar_crash_reversal(raw.head(end), -575, -0.5, 60, bar_interval_seconds=3600)
        assert prefix['dollar_bar_crash_reversal_position'].equals(actual['dollar_bar_crash_reversal_position'].head(end))


def test_late_censored_opportunity_preserves_nullable_label_schema():
    _, _, _, data = _prepared()
    raw = pl.read_parquet(FIXTURE, n_rows=1800)
    rising = (pl.col('close') > pl.col('open'))
    end = raw.filter(rising & rising.shift(1))['datetime'][-1]
    raw = raw.filter(pl.col('datetime') <= end)
    context = data['_allocation_context']
    policy = replace(context.policy, take_profit_bps_column=None, stop_loss_bps_column=None,
                     barrier_spec=None, max_holding_seconds=None)
    context = replace(context, raw_train=raw, policy=policy, barriers=None)
    labels = training_opportunities(context, folds=2, validation_fraction=0.2, include_primary_proba=False)
    rows = labels.rows
    assert rows.height > 100
    assert rows['unavailable_reason'].head(100).null_count() == 100
    assert rows['unavailable_reason'][-1] == 'open_at_opening_block_end'
    assert rows['available'][-1] is False
    assert rows['label_available_ns'][-1] is None
    assert rows['net_return_on_entry_notional'][-1] is None
    completed = rows.filter(pl.col('available'))
    assert completed.height == rows.height - 1
    np.testing.assert_allclose(completed['net_return_on_entry_notional'], completed['net_pnl'] / completed['entry_notional'])
    assert labels.features.height == rows.height
    assert rows['opportunity_id'].n_unique() == rows.height
