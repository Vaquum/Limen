import copy
from datetime import date
import math
import numpy as np
from unittest.mock import patch
from types import SimpleNamespace

import polars as pl
import pytest

from limen.experiment._fold_evidence import read_fold_context
from limen.experiment._fold_fingerprint import fingerprint, membership
from limen.inference import Trainer
from limen.inference._fold_metrics import scalar_matches
from limen.inference.errors import ReconstructionError
pytest_plugins = ['tests.fold_components']


@pytest.mark.parametrize('kind', ['anchored', 'rolling', 'continuous', 'custom', 'formed', 'parallel', 'signed', 'reordered', 'duplicates', 'companion'])
def test_selected_fold_reconstruction(fold_campaigns, kind):
    campaign = fold_campaigns[kind]
    sensor = campaign.sensor
    witness = campaign.record['folds'][campaign.fold]['fold_promotion']
    assert sensor.fold == campaign.fold
    assert sensor.n_folds == campaign.metadata['split_walk_forward']['n_folds']
    assert sensor.fold_provenance['fold_witness'] == witness
    assert sensor.fold_provenance['results'] == campaign.record['folds'][campaign.fold]['results']
    assert sensor.provenance_digest == read_fold_context(campaign.metadata, campaign.record, campaign.fold)['provenance_digest']
    copied = sensor.fold_provenance
    copied['fold_witness']['fold'] = -1
    assert sensor.fold_provenance['fold_witness']['fold'] == campaign.fold
    assert sensor.fitting_cutoff <= sensor.selection_cutoff
    assert sensor.selection_cutoff.timestamp() * 1e9 == campaign.metadata['fold_promotion']['selection_cutoff_ns']
    if kind in ('anchored', 'rolling'):
        fit, validation = witness['raw']['fit'], witness['raw']['validation']
        assert fit['segments'] == ([[0, 240], [243, 259]] if kind == 'anchored' else [[144, 240], [243, 252]])
        assert validation['segments'] == ([[263, 288], [291, 332]] if kind == 'anchored' else [[256, 288], [291, 332]])
        assert fit['rows'] == (256 if kind == 'anchored' else 105)
        assert validation['rows'] == (66 if kind == 'anchored' else 73)
        assert witness['fitted_threshold'] == 0.5
    if kind == 'parallel':
        assert campaign.metadata['fold_promotion']['origin'] == 'parallel'
    if kind == 'signed':
        assert sensor.prediction_mode == 'target_exposure'
        assert campaign.record['folds'][0]['learning_binding']
    if kind == 'custom':
        other = fold_campaigns['companion']
        assert other.record['round_id'] == campaign.record['round_id']
        assert other.metadata['manifest_id'] == campaign.metadata['manifest_id']
        assert other.sensor.provenance_digest != sensor.provenance_digest


@pytest.mark.parametrize('mutation', ['lean', 'witness', 'version', 'source', 'ratio', 'raw', 'retained', 'threshold', 'metric', 'binding', 'model', 'count', 'competing', 'csv', 'other_unavailable', 'selected_unavailable', 'identity'])
def test_witness_validation_stages(fold_campaigns, mutation):
    campaign = fold_campaigns['anchored']
    trainer = Trainer(campaign.path, data=campaign.bars)
    pid = campaign.record['round_id']
    entry = trainer._round_data[pid]
    witness = entry['folds'][campaign.fold]['fold_promotion']
    if mutation == 'lean':
        trainer._metadata['recording']['fold_details'] = False
    if mutation == 'identity':
        trainer._metadata['manifest_id'] = None
    if mutation == 'witness':
        entry.pop('fold_promotion')
    if mutation == 'version':
        witness['version'] = 2
    if mutation == 'source':
        trainer._data = campaign.bars.slice(1)
    if mutation == 'ratio':
        trainer._metadata['walk_forward_validation_ratio'] = [8, 1]
    if mutation == 'raw':
        witness['raw']['fit']['segments'][0][1] -= 1
    if mutation == 'retained':
        witness['retained']['test']['row_digest'] = '0' * 64
    if mutation == 'threshold':
        witness['fitted_threshold'] += 1e-3
    if mutation == 'metric':
        entry['folds'][campaign.fold]['results']['missing_replay_metric'] = 1
    if mutation == 'binding':
        entry['folds'][campaign.fold]['learning_binding'] = {'changed': True}
    if mutation == 'model':
        witness['inference_model']['output_contract']['kind'] = 'continuous'
    if mutation == 'count':
        witness['n_folds'] += 1
    if mutation == 'competing':
        trainer._round_data['competitor'] = copy.deepcopy(entry)
        trainer._round_data['competitor']['fold_promotion']['campaign_source_digest'] = '0' * 64
    if mutation in ('other_unavailable', 'selected_unavailable'):
        target = 0 if mutation == 'other_unavailable' else campaign.fold
        entry['folds'][target]['fold_promotion'] = {'version': 1, 'unavailable_reason': {'code': 'inference_model', 'detail': 'no retained fitted model'}}
        if mutation == 'other_unavailable':
            assert trainer.train([pid], fold=campaign.fold)[0].fold == campaign.fold
            return
    if mutation == 'csv':
        with patch('limen.inference.trainer.load_original_log', side_effect=AssertionError('must not read aggregate')):
            Trainer(campaign.path, data=campaign.bars)
        return
    before_fit = mutation in ('lean', 'witness', 'version', 'source', 'ratio', 'raw', 'count', 'competing', 'identity')
    if before_fit:
        guard = patch('limen.experiment.manifest_core._apply_scaler', side_effect=AssertionError('preprocessing fitted too early'))
    elif mutation == 'retained':
        guard = patch('limen.experiment.manifest_core.MLManifest.run_model', side_effect=AssertionError('model fitted before retained validation'))
    else:
        guard = patch.object(trainer, '_validate_metrics', side_effect=AssertionError('aggregate used'))
    with guard, pytest.raises(ReconstructionError, match=f'Permutation {pid}, fold {campaign.fold}'):
        trainer.train([pid], fold=campaign.fold)


def test_ambiguous_selected_rows_and_incomplete_model_decline(fold_campaigns):
    from limen.experiment._fold_evidence import model_witness
    from limen.experiment._fold_fingerprint import FoldEvidenceError
    from limen.experiment._fold_partitions import raw

    rows = fold_campaigns['anchored'].history.head(4)
    duplicate = pl.concat([rows, rows.head(1)])
    state = {'trial': {}}
    raw(SimpleNamespace(_fold_capture=state), duplicate, [duplicate, rows.clear(), rows])
    assert state['unavailable']['unavailable_reason']['code'] == 'row_membership'
    with pytest.raises(FoldEvidenceError, match='native prediction_mode'):
        model_witness(SimpleNamespace(predict=lambda data: data))


@pytest.mark.parametrize('extension', ['unchanged', 'numpy', 'uncarried_reverse', 'carried_reverse'])
def test_extension_rows_require_recoverable_carriers(fold_campaigns, extension):
    from limen.experiment import _fold_partitions as parts

    selected = fold_campaigns['anchored'].history.head(6).select('datetime', 'open', 'close')
    frames = [selected[index:index + 2] for index in (0, 2, 4)]
    manifest = SimpleNamespace(_fold_capture={'trial': {}}, split_walk_forward=SimpleNamespace(n_folds=2), target_column='close')
    parts.raw(manifest, selected, frames)
    for frame in frames:
        parts.bars(manifest, frame, frame, frame['datetime'].to_list())
    if extension == 'carried_reverse':
        frames[0] = frames[0].reverse()
    data = {}
    for name, frame in zip(('train', 'val', 'test'), frames, strict=True):
        data['x_' + name], data['y_' + name] = frame.select('open'), frame['close']
    if extension == 'uncarried_reverse':
        data['x_train'], data['y_train'] = data['x_train'].reverse(), data['y_train'].reverse()
    if extension == 'numpy':
        for name in ('train', 'val', 'test'):
            data['x_' + name] = data['x_' + name].to_numpy()
            data['y_' + name] = data['y_' + name].to_numpy().reshape(-1, 1)
    result = parts.retained(manifest, frames, data)['_fold_partitions']
    if extension == 'uncarried_reverse':
        assert result['unavailable_reason']['code'] == 'row_membership'
    else:
        assert 'unavailable_reason' not in result
        if extension == 'carried_reverse':
            assert result['raw']['fit']['row_digest'] != result['retained']['fit']['row_digest']


def test_parallel_reuses_frozen_campaign_fingerprint(fold_campaigns):
    from limen.experiment import _fold_evidence as evidence

    item = fold_campaigns['parallel']
    campaign = {key: value for key, value in item.metadata['fold_promotion'].items() if key != 'origin'}
    with patch.object(evidence, 'source', side_effect=AssertionError('original source rehashed in worker')):
        copied = evidence.start(item.sensor._get_manifest(), item.bars, campaign)
    assert copied == campaign and copied is not campaign


def test_fitted_float32_predictions_keep_numeric_row_parity(fold_campaigns):
    from limen.inference._fold_metrics import stable_model

    item = fold_campaigns['continuous']
    sensor = item.sensor
    prepared, _ = sensor._get_manifest().sensor_input_prep(item.history.head(600), sensor._fitted_params, sensor.round_params)
    inputs = prepared.drop('datetime').drop_nulls().to_numpy()
    weights = sensor._model.model.coef_.astype(np.float32)
    bias = np.float32(sensor._model.model.intercept_)

    class Float32Model(type(sensor._model)):
        def predict(self, data):
            return {'_preds': np.asarray(data['x_test'], dtype=np.float32) @ weights + bias}

    stable_model(Float32Model(), {'x_train': inputs[:200], 'x_val': inputs[200:300], 'x_test': inputs[300:]})


def test_fold_fingerprint_codec(fold_campaigns):
    frame = fold_campaigns['anchored'].history.head(2)
    assert fingerprint(frame) == '4d17b350e56611cd6cde9fb7059098f64a665df3b99098b45ca625ea993ae5bf'
    assert membership([[0], [1]]) == '0144ca736aaf96da235bb3482b5990551e7a9a00863470a0073324fe4a0ff45f'
    for unit in ('ms', 'us', 'ns'):
        assert fingerprint(frame) == fingerprint(frame.with_columns(pl.col('datetime').dt.cast_time_unit(unit)))
    # Protocol controls are scalar encodings, not fabricated market observations.
    controls = pl.DataFrame({'signed': pl.Series([-1, 0], dtype=pl.Int8), 'float': [None, -0.0], 'bool': [True, False], 'text': ['ä', ''], 'date': [date(2020, 1, 1), date(2020, 1, 2)]})
    assert fingerprint(controls) != fingerprint(controls.with_columns(pl.lit(0.0).alias('float')))
    assert fingerprint(controls) != fingerprint(controls.with_columns(pl.Series('float', [None, 0.0])))
    nonfinite = [fingerprint(controls.with_columns(pl.lit(value).alias('float'))) for value in (math.nan, math.inf, -math.inf)]
    assert len(set(nonfinite)) == 3
    for left, right, expected in [(None, None, True), (None, 0, False), (math.nan, math.nan, True), (math.inf, -math.inf, False), (True, 1, False), (1.0, 1.00001, False)]:
        assert scalar_matches(left, right, True) is expected


@pytest.mark.parametrize('fold,message', [(None, 'requires train'), (True, 'literal integer'), (-1, 'out of range'), (5, 'out of range')])
def test_fold_argument_errors(fold_campaigns, fold, message):
    item = fold_campaigns['anchored']
    with pytest.raises(ValueError, match=message):
        Trainer(item.path, data=item.bars).train([item.record['round_id']], fold=fold)
