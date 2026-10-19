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
    return config


def _prepared(ml=False):
    config = _config(ml)
    manifest = CompiledSFD(config).manifest()
    raw = pl.read_parquet(FIXTURE, n_rows=480)
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    return manifest, raw, params, manifest.prepare_data(raw, params)


@pytest.mark.parametrize('ml', [False, True])
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
        assert opened['label_available_ns'].le(block['end_ns']).all()
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
    path = tmp_path / 'allocation.yaml'
    with path.open('w') as stream:
        YAML().dump(config, stream)
    for command in (['validate', str(path)], ['run', str(path), '--dry-run']):
        result = CliRunner().invoke(cli, command)
        assert result.exit_code == 0, result.output
    manifest, raw, params, _ = _prepared()
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
        dollar_bar_crash_reversal(raw, -575, -0.5, 60, bar_interval_seconds=900)
