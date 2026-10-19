import copy
from datetime import date
import json
import math
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from limen.experiment._fold_evidence import read_fold_context
from limen.experiment._fold_fingerprint import fingerprint, membership
from limen.inference import Trainer
from limen.inference._fold_metrics import scalar_matches
from limen.inference.errors import ReconstructionError
from tests.fold_components import FIXTURE, fold_campaigns


@pytest.mark.parametrize('kind', ['anchored', 'rolling', 'continuous', 'custom', 'formed', 'parallel', 'signed', 'reordered', 'companion'])
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


@pytest.mark.parametrize('mutation', ['lean', 'witness', 'version', 'source', 'ratio', 'raw', 'retained', 'threshold', 'metric', 'binding', 'model', 'count', 'competing', 'csv'])
def test_witness_validation_stages(fold_campaigns, mutation):
    campaign = fold_campaigns['anchored']
    trainer = Trainer(campaign.path, data=campaign.bars)
    pid = campaign.record['round_id']
    entry = trainer._round_data[pid]
    witness = entry['folds'][campaign.fold]['fold_promotion']
    if mutation == 'lean': trainer._metadata['recording']['fold_details'] = False
    if mutation == 'witness': entry.pop('fold_promotion')
    if mutation == 'version': witness['version'] = 2
    if mutation == 'source': trainer._data = campaign.bars.slice(1)
    if mutation == 'ratio': trainer._metadata['walk_forward_validation_ratio'] = [8, 1]
    if mutation == 'raw': witness['raw']['fit']['segments'][0][1] -= 1
    if mutation == 'retained': witness['retained']['test']['row_digest'] = '0' * 64
    if mutation == 'threshold': witness['fitted_threshold'] += 1e-3
    if mutation == 'metric': entry['folds'][campaign.fold]['results']['missing_replay_metric'] = 1
    if mutation == 'binding': entry['folds'][campaign.fold]['learning_binding'] = {'changed': True}
    if mutation == 'model': witness['inference_model']['output_contract']['kind'] = 'continuous'
    if mutation == 'count': witness['n_folds'] += 1
    if mutation == 'competing':
        trainer._round_data['competitor'] = copy.deepcopy(entry)
        trainer._round_data['competitor']['fold_promotion']['campaign_source_digest'] = '0' * 64
    if mutation == 'csv':
        with patch('limen.inference.trainer.load_original_log', side_effect=AssertionError('must not read aggregate')):
            Trainer(campaign.path, data=campaign.bars)
        return
    before_fit = mutation in ('lean', 'witness', 'version', 'source', 'ratio', 'raw', 'count', 'competing')
    with patch('limen.experiment.manifest_core._apply_scaler', side_effect=AssertionError('preprocessing fitted too early')) if before_fit else patch.object(trainer, '_validate_metrics', side_effect=AssertionError('aggregate used')):
        with pytest.raises(ReconstructionError, match=f'Permutation {pid}, fold {campaign.fold}'):
            trainer.train([pid], fold=campaign.fold)


def test_fold_fingerprint_codec():
    frame = pl.read_parquet(FIXTURE, n_rows=2)
    assert fingerprint(frame) == '4d17b350e56611cd6cde9fb7059098f64a665df3b99098b45ca625ea993ae5bf'
    assert membership([[0], [1]]) == '0144ca736aaf96da235bb3482b5990551e7a9a00863470a0073324fe4a0ff45f'
    assert fingerprint(frame) == fingerprint(frame.with_columns(pl.col('datetime').dt.cast_time_unit('ns')))
    # Protocol controls are scalar encodings, not fabricated market observations.
    controls = pl.DataFrame({'signed': pl.Series([-1, 0], dtype=pl.Int8), 'float': [None, -0.0], 'bool': [True, False], 'text': ['ä', ''], 'date': [date(2020, 1, 1), date(2020, 1, 2)]})
    assert fingerprint(controls) != fingerprint(controls.with_columns(pl.lit(0.0).alias('float')))
    for left, right, expected in [(None, None, True), (None, 0, False), (math.nan, math.nan, True), (math.inf, -math.inf, False), (True, 1, False), (1.0, 1.00001, False)]:
        assert scalar_matches(left, right, True) is expected


@pytest.mark.parametrize('fold,message', [(None, 'requires train'), (True, 'literal integer'), (-1, 'out of range'), (5, 'out of range')])
def test_fold_argument_errors(fold_campaigns, fold, message):
    item = fold_campaigns['anchored']
    with pytest.raises(ValueError, match=message):
        Trainer(item.path, data=item.bars).train([item.record['round_id']], fold=fold)
