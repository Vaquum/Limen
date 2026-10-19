import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from limen import Cohort
from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.inference import Trainer
from limen.inference.errors import ReconstructionError
from limen.sfd.reference_architecture import LogRegBinary
from limen.yaml import CompiledSFD, build_search_strategy

pytest_plugins = ['tests.fold_components']


class Proxy:
    def __init__(self, sensor, **changes):
        self.sensor, self.changes = sensor, changes

    def __getattr__(self, name):
        return self.changes[name] if name in self.changes else getattr(self.sensor, name)


class VoteOnly(LogRegBinary):
    inference_output = ('binary', False)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict(self, data):
        return {'_preds': self.model.predict(data)['_preds']}


class Undeclared(VoteOnly):
    inference_output = None


@pytest.fixture(scope='session')
def cohort_assets(fold_campaigns, tmp_path_factory):
    from tests.fold_components import FULL

    started = time.perf_counter()
    additional = {}
    with threadpool_limits(limits=1):
        for kind in ('anchored', 'formed', 'parallel', 'custom'):
            item = fold_campaigns[kind]
            other = 2 if kind == 'anchored' else 1
            additional[kind] = Trainer(item.path, data=item.bars).train([item.record['round_id']], fold=other)[0]
        item = fold_campaigns['formed']
        reference = copy.deepcopy(item.config)
        manifest = reference['sfd']['manifest']
        manifest.pop('split_walk_forward')
        manifest['split_dates'] = copy.deepcopy(fold_campaigns['ordinary'].config['sfd']['manifest']['split_dates'])
        output = tmp_path_factory.mktemp('ordinary-formed')
        loop = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=item.bars, experiment_dir=output,
                                       yaml_reference=reference, search_strategy=build_search_strategy(reference))
        loop.run('ordinary_formed', n_permutations=1, prep_each_round=True, progress_bar=False,
                 recording=FULL, record_execution=True, record_model_outputs=True)
        trainer = Trainer(output, data=item.bars)
        pid = next(iter(trainer._round_data))
        ordinary = SimpleNamespace(path=output, record={'round_id': pid}, sensor=trainer.train([pid])[0])
    elapsed = time.perf_counter() - started
    Path('build').mkdir(exist_ok=True)
    Path('build/fold-cohort-profile.json').write_text(json.dumps({'extra_setup_seconds': elapsed, 'additional_fold_refits': 4, 'ordinary_formed_runs': 1, 'ordinary_formed_refits': 1}, indent=2))
    return additional, ordinary


def _select(item, fold=None, path=None):
    return (item.path if path is None else path, item.record['round_id'], item.fold if fold is None else fold)


def _same_prediction(left, right):
    assert (left.datetime, left.prediction, left.reason) == (right.datetime, right.prediction, right.reason)
    assert left.probability == pytest.approx(right.probability) if right.probability is not None else left.probability is None


def _copy(item, destination, change=None):
    shutil.copytree(item.path, destination)
    if change is not None:
        metadata, record = copy.deepcopy(item.metadata), copy.deepcopy(item.record)
        change(metadata, record)
        (destination / 'metadata.json').write_text(json.dumps(metadata))
        _round_storage.round_path(destination).write_bytes(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0))
    return destination


@pytest.mark.parametrize('kind', ['anchored', 'formed', 'parallel', 'custom'])
def test_distinct_fold_members(fold_campaigns, cohort_assets, tmp_path, kind):
    item = fold_campaigns[kind]
    first, second = item.sensor, cohort_assets[0][kind]
    copied = _copy(item, tmp_path / 'copy')
    selections = [_select(item), _select(item, second.fold, copied)]
    with patch.object(Trainer, 'train', side_effect=AssertionError('selection/binding must not fit')):
        cohort = Cohort(rounds=selections)
        assert cohort.cohort_id is None and cohort.aggregation_mode is None and cohort.supports_probabilities is None
        assert cohort.permutation_ids is cohort.experiment_dir is cohort.metadata is cohort.available_permutation_ids is None
        assert cohort.manifest_id == first.manifest_id
        assert set(cohort.source_metadata) == {item.path.resolve(), copied.resolve()}
        assert cohort.member_ids == [(first.manifest_id, first.permutation_id, first.fold, first.provenance_digest),
                                     (second.manifest_id, second.permutation_id, second.fold, second.provenance_digest)]
        cohort.set_members([Proxy(second), Proxy(first)])
        reversed_cohort = Cohort(rounds=list(reversed(selections)))
        reversed_cohort.set_members([first, second])
    assert cohort.cohort_id == reversed_cohort.cohort_id
    assert cohort.supports_probabilities is True and cohort.aggregation_mode == 'probability_weighted'
    rows = sorted([['fold', first.manifest_id, first.permutation_id, first.fold, first.provenance_digest],
                   ['fold', second.manifest_id, second.permutation_id, second.fold, second.provenance_digest]], key=lambda row: row[3])
    payload = {'architecture_id': cohort.architecture_id, 'aggregation_mode': 'probability_weighted', 'members': rows}
    assert cohort.cohort_id == 'sha256:' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    history = item.history.head(600)
    prediction = cohort.predict(history)
    assert prediction.reason is None
    assert prediction.probability == pytest.approx(np.mean([first.predict(history).probability, second.predict(history).probability]))
    _same_prediction(prediction, cohort.predict_all(history)[-1])
    assert first.fold != second.fold and first.provenance_digest != second.provenance_digest
    single = Cohort(rounds=[_select(item)])
    assert single.experiment_dir == item.path.resolve() and single.metadata == item.metadata
    assert single.available_permutation_ids and single.permutation_ids is None
    single.set_members([first])
    from limen.experiment._holdout_freeze import members
    with pytest.raises(ValueError, match='Holdout Cohort member bindings disagree'):
        members(single)
    same = Cohort(rounds=[_select(item, path=copied)])
    same.set_members([first])
    assert same.cohort_id == single.cohort_id
    alternate = Cohort(rounds=[_select(item, second.fold)])
    alternate.set_members([second])
    assert alternate.cohort_id != single.cohort_id


@pytest.mark.parametrize('fault', ['wrong_fold', 'digest', 'n_folds', 'duplicate_bound', 'duplicate_selected', 'conflict', 'raw', 'version', 'declaration', 'unqualified', 'yaml_only', 'ordinary_fold', 'missing', 'json', 'range_before_id', 'unknown_kind', 'continuous', 'signed', 'count', 'architecture', 'hidden_fold', 'missing_round', 'metadata_shape', 'model_schema', 'n_folds_schema'])
def test_selected_artifact_binding(fold_campaigns, cohort_assets, tmp_path, fault):
    item = fold_campaigns['custom'] if fault == 'conflict' else fold_campaigns['anchored']
    first, second = item.sensor, cohort_assets[0]['custom' if fault == 'conflict' else 'anchored']
    if fault in ('missing', 'missing_round', 'metadata_shape'):
        path = tmp_path / 'absent' if fault == 'missing' else _copy(item, tmp_path / 'invalid-source')
        if fault == 'missing_round':
            _round_storage.round_path(path).unlink()
        elif fault == 'metadata_shape':
            (path / 'metadata.json').write_text('[]')
        with pytest.raises(ReconstructionError if fault == 'metadata_shape' else FileNotFoundError):
            Cohort(rounds=[_select(item, path=path)])
        return
    if fault in ('continuous', 'signed'):
        with pytest.raises(ValueError, match='Cohort requires declared binary'):
            Cohort(rounds=[_select(fold_campaigns[fault])])
        return
    if fault in ('duplicate_selected', 'conflict'):
        path = _copy(item, tmp_path / 'alias') if fault == 'duplicate_selected' else fold_campaigns['companion'].path
        with pytest.raises(ValueError, match='Cohort duplicate' if fault == 'duplicate_selected' else 'Cohort evidence conflict'):
            Cohort(rounds=[_select(item), _select(item, path=path)])
        return
    if fault in ('raw', 'version', 'declaration', 'yaml_only', 'unknown_kind', 'json', 'model_schema', 'n_folds_schema'):
        def change(metadata, record):
            if fault == 'raw':
                record['folds'][item.fold]['fold_promotion']['raw']['fit']['row_digest'] = '0' * 64
            if fault == 'version':
                record['folds'][item.fold]['fold_promotion']['version'] = 2
            if fault == 'declaration':
                metadata['split_walk_forward']['n_folds'] += 1
            if fault == 'yaml_only':
                metadata.pop('split_walk_forward')
            if fault == 'n_folds_schema':
                metadata['split_walk_forward']['n_folds'] = float(metadata['split_walk_forward']['n_folds'])
            if fault == 'model_schema':
                record['folds'][item.fold]['fold_promotion']['inference_model'].pop('class')
            if fault == 'unknown_kind':
                record['folds'][item.fold]['fold_promotion']['inference_model']['output_contract'] = None
        path = _copy(item, tmp_path / 'changed', change)
        if fault == 'json':
            (path / 'metadata.json').write_text('{broken')
        if fault == 'yaml_only':
            with patch.object(Cohort, '_load_round_entries', side_effect=AssertionError('read entries before unqualified rejection')), pytest.raises(ValueError, match=r'walk-forward rounds cannot be promoted.*supply fold'):
                Cohort(rounds=[(path, item.record['round_id'])])
        else:
            expected = ValueError if fault == 'unknown_kind' else ReconstructionError
            with pytest.raises(expected, match='Cohort requires declared binary' if fault == 'unknown_kind' else 'Cohort source'):
                Cohort(rounds=[_select(item, path=path)])
        return
    if fault in ('unqualified', 'ordinary_fold', 'range_before_id'):
        with patch.object(Cohort, '_load_round_entries', side_effect=AssertionError('invalid selection read round entries')), pytest.raises(ValueError, match='Cohort'):
            if fault == 'ordinary_fold':
                ordinary = fold_campaigns['ordinary']
                Cohort(rounds=[(ordinary.path, ordinary.record['round_id'], 0)])
            elif fault == 'range_before_id':
                Cohort(rounds=[(item.path, 'unknown', first.n_folds)])
            else:
                Cohort(rounds=[(item.path, item.record['round_id'])])
        return
    cohort = Cohort(rounds=[_select(item)] if fault == 'wrong_fold' else [_select(item), _select(item, second.fold)])
    if fault in ('count', 'architecture', 'hidden_fold'):
        members = [first] if fault == 'count' else [Proxy(first, round_params={**first.round_params, 'architecture': 'wrong'}) if fault == 'architecture' else Proxy(first, fold=None), second]
        with pytest.raises(ValueError, match='Cohort'):
            cohort.set_members(members)
        assert cohort.cohort_id is None and cohort.supports_probabilities is None
        return
    replacements = {'digest': Proxy(first, provenance_digest='0' * 64), 'n_folds': Proxy(first, n_folds=99), 'wrong_fold': second, 'duplicate_bound': first}
    members = [second] if fault == 'wrong_fold' else [first, first] if fault == 'duplicate_bound' else [replacements[fault], second]
    message = 'Cohort unknown or wrong-fold' if fault == 'wrong_fold' else 'Cohort duplicate bound' if fault == 'duplicate_bound' else 'Cohort selected evidence mismatch'
    with pytest.raises(ValueError, match=message):
        cohort.set_members(members)
    assert cohort.cohort_id is None and cohort.aggregation_mode is None


@pytest.mark.parametrize('kind', ['native', 'formed', 'vote'])
@pytest.mark.parametrize('failure', ['none', 'feature', 'model'])
def test_mixed_guard_and_ordinary_identity(fold_campaigns, cohort_assets, kind, failure):
    item = fold_campaigns['formed' if kind == 'formed' else 'anchored']
    ordinary = cohort_assets[1] if kind == 'formed' else fold_campaigns['ordinary']
    fold_sensor, native = copy.deepcopy(item.sensor), copy.deepcopy(ordinary.sensor)
    if kind == 'vote':
        native._model = VoteOnly(native._model)
    selections = [_select(item), (ordinary.path, ordinary.record['round_id'])]
    cohort = Cohort(rounds=selections)
    cohort.set_members([native, fold_sensor])
    assert cohort.aggregation_mode == ('majority_vote' if kind == 'vote' else 'probability_weighted')
    same = Cohort(rounds=list(reversed(selections)))
    same.set_members([fold_sensor, native])
    assert same.cohort_id == cohort.cohort_id
    manifest = fold_sensor._get_manifest()
    if failure == 'feature':
        context = patch.object(manifest, '_prepare_column_shims', side_effect=ValueError('feature failure'))
    elif failure == 'model':
        context = patch.object(fold_sensor._model, 'predict', side_effect=ValueError('model failure'))
    else:
        context = patch.object(manifest, 'val_predict_guard', False)
    history = item.history.head(600)
    with context:
        rows = cohort.predict_all(history)
        assert rows
        _same_prediction(cohort.predict(history), rows[-1])
        assert any(row.datetime >= fold_sensor.selection_cutoff for row in rows)
        for row in rows:
            if row.datetime < fold_sensor.selection_cutoff:
                assert row.reason == 'inside-training-window' and row.prediction is row.probability is None
        if failure == 'none':
            assert rows[-1].reason is None and rows[-1].prediction is not None
    if kind == 'formed':
        assert len(rows) < len(history)
    original = Cohort(experiment_log_path=str(ordinary.path), permutation_ids=[ordinary.record['round_id']])
    original.set_members([ordinary.sensor])
    legacy_rounds = Cohort(rounds=[(ordinary.path, ordinary.record['round_id'])])
    legacy_rounds.set_members([ordinary.sensor])
    assert legacy_rounds.cohort_id == original.cohort_id
    assert legacy_rounds.member_ids == original.member_ids
    with pytest.raises(ValueError, match='fold-qualified'):
        Cohort.from_python([fold_sensor], map_output=lambda value: value, mapped_output_id='binary', selection_cutoff=fold_sensor.selection_cutoff)


@pytest.mark.parametrize('selection,exception', [([], ValueError), ([('missing', 'id', True)], ValueError), ([('missing', 'id', -1)], ValueError), ([('missing', 'id', np.int64(0))], ValueError), ([('missing', 'id', None)], ValueError), ([('missing', ' ')], ValueError), ([('missing', 'id', 0, 'extra')], ValueError), ([(object(), 'id')], TypeError), ('invalid', TypeError)])
def test_fold_selection_shape_precedes_files(selection, exception):
    with pytest.raises(exception, match='Cohort'):
        Cohort(rounds=selection)


@pytest.mark.parametrize('fault', ['source', 'formation', 'campaign', 'split', 'ordinary_output', 'interleaved'])
def test_fold_compatibility(fold_campaigns, cohort_assets, tmp_path, fault):
    item = fold_campaigns['anchored']
    ordinary = fold_campaigns['ordinary']
    if fault == 'interleaved':
        second = cohort_assets[0]['anchored']
        cohort = Cohort(rounds=[_select(item), (ordinary.path, ordinary.record['round_id']), _select(item, second.fold)])
        cohort.set_members([second, ordinary.sensor, item.sensor])
        assert [len(member) for member in cohort.member_ids] == [4, 2, 4]
        assert [member.fold if hasattr(member, 'fold') else None for member in cohort._members] == [item.fold, None, second.fold]
        return
    if fault == 'ordinary_output':
        cohort = Cohort(rounds=[_select(item), (ordinary.path, ordinary.record['round_id'])])
        model = Undeclared(copy.deepcopy(ordinary.sensor._model))
        assert type(item.sensor._model).inference_output == ('binary', True)
        with pytest.raises(ValueError, match='Cohort requires declared binary') as error:
            cohort.set_members([item.sensor, Proxy(ordinary.sensor, _model=model)])
        assert str((ordinary.sensor.manifest_id, ordinary.sensor.permutation_id)) in str(error.value)
        assert cohort.cohort_id is None and cohort.supports_probabilities is None
        return
    if fault == 'campaign':
        first = fold_campaigns['custom']
        second = fold_campaigns['companion']
        with pytest.raises(ValueError, match='incompatible fold campaign'):
            Cohort(rounds=[_select(first), _select(second, 1)])
        return
    if fault == 'split':
        def change_split(metadata, record):
            metadata['split_walk_forward']['anchored'] = False
            metadata['yaml_reference']['sfd']['manifest']['split_walk_forward']['anchored'] = False
        path = _copy(item, tmp_path / 'different-split', change_split)
        with pytest.raises(ValueError, match='incompatible fold campaign'):
            Cohort(rounds=[_select(item), _select(item, 2, path)])
        return
    if fault == 'formation':
        with pytest.raises(ValueError, match='incompatible data source, architecture or effective bar formation'):
            Cohort(rounds=[_select(item), _select(fold_campaigns['formed'])])
        return
    def change(metadata, record):
        metadata['yaml_reference']['sfd']['manifest']['data_source']['params']['file_path_or_url'] = 'different.parquet'
    path = _copy(ordinary, tmp_path / 'different-source', change)
    with pytest.raises(ValueError, match='incompatible data source'):
        Cohort(rounds=[_select(item), (path, ordinary.record['round_id'])])
