import gzip
from limen.experiment import _round_storage as _storage
import copy
import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from limen.cli.commands.run import run_experiment
from limen.cohort import Cohort
from limen.data import HistoricalData
from limen.inference import Trainer
from limen.inference.errors import ReconstructionError
from limen.inference.sensor import Sensor

_FIXTURE = Path(__file__).parent / 'fixtures' / 'spot_15m_20250101_20250531.parquet'
_FIXTURE_SHA = 'eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a'


@dataclass
class _Campaign:
    directory: Path
    ids: list[str]
    sensors: list[Sensor]


def _yaml(directory, seed, *, narrowed=False):
    dates = ('02', '05', '06', '07') if narrowed else ('01', '04', '05', '06')
    train_start, val_start, test_start, test_end = dates
    return dedent(f'''\
        schema_version: "1.0"
        metadata:
          name: recorded_cohort
          mode: development
        sfd:
          manifest:
            type: ml
            data_source:
              method: limen.data.HistoricalData.get_any_file
              params:
                file_path_or_url: "{_FIXTURE.resolve()}"
            split_dates:
              train_start: "2025-01-{train_start}"
              train_end: "2025-01-{val_start}"
              val_start: "2025-01-{val_start}"
              val_end: "2025-01-{test_start}"
              test_start: "2025-01-{test_start}"
              test_end: "2025-01-{test_end}"
            indicators:
              - func: limen.indicators.roc
                params:
                  period: 1
            target:
              name: quantile_flag
              class: limen.targets.QuantileBinaryTarget
              fit_params:
                source_column: roc_1
                quantile: 0.5
              transform_params:
                shift: -1
            scaler:
              from_params: scaler_type
            reference_architecture: limen.sfd.reference_architecture.logreg_binary
          params:
            scaler_type: [logreg]
            C: {'[1.0]' if narrowed else '[0.1, 1.0, 10.0]'}
            random_state: [42]
            max_iter: [100]
        uel:
          n_permutations: {1 if narrowed else 3}
          search_strategy:
            type: grid
            seed: {seed}
          checkpoint_interval: 1
          record_execution: true
          output_format: csv
          output_path: "{directory.name}"
        ''')


@pytest.fixture(scope='module')
def cohort_campaigns(tmp_path_factory):
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _FIXTURE_SHA
    recorded = HistoricalData().get_any_file(str(_FIXTURE))
    root = tmp_path_factory.mktemp('recorded_cohort')
    campaigns = []
    with patch('requests.sessions.Session.request', side_effect=AssertionError('use recorded market data')):
        for name, seed, narrowed in (('first', 42, False), ('second', 43, False), ('descendant', 43, True)):
            directory = root / 'results/dev' / name
            yaml_path = root / f'{name}.yaml'
            yaml_path.write_text(_yaml(directory, seed, narrowed=narrowed))
            assert run_experiment(yaml_path, results_base=root, progress_bar=False)
            ids = pl.read_csv(directory / 'results.csv')['id'].to_list()
            sensors = Trainer(directory, data=recorded).train(ids)
            campaigns.append(_Campaign(directory, ids, sensors))
    return campaigns, recorded


def _after_training(recorded):
    return recorded.filter(pl.col('datetime').dt.date().cast(pl.String) == '2025-01-08')


@pytest.mark.parametrize('source_form', ['legacy', 'rounds'])
def test_cohort_rejects_python_sensor(cohort_campaigns, source_form):
    campaigns, recorded = cohort_campaigns
    first = campaigns[0]
    ids = first.ids[:2]
    if source_form == 'legacy':
        cohort = Cohort(experiment_log_path=str(first.directory), permutation_ids=ids)
    else:
        cohort = Cohort(rounds=[(first.directory, pid) for pid in ids])
    raw = _after_training(recorded)
    metadata = raw.head(1).select('datetime')
    cutoff = metadata['datetime'][0]
    python_sensor = Sensor.from_python(
        fitted_state=first.sensors[1], params={},
        infer=lambda history, state, params: (metadata, [(1.0, None)]),
        map_output=lambda output: output, state_id='caller-composition',
        fitting_cutoff=cutoff, selection_cutoff=cutoff,
    )
    message = 'Cohort cannot bind Python-origin Sensors; use Cohort.from_python.'
    with pytest.raises(ValueError, match=f'^{message.replace(".", "[.]")}$'):
        cohort.set_members([first.sensors[0], python_sensor])
    assert cohort.cohort_id is None
    for method in (cohort.predict, cohort.predict_all):
        with pytest.raises(RuntimeError, match='no bound members'):
            method(raw)


def test_python_cohort_mixes_compiled_binary_sensor(cohort_campaigns):
    campaigns, recorded = cohort_campaigns
    native = campaigns[0].sensors[0]
    identity = (native.manifest_id, native.permutation_id, native.trade_contract)

    def features(history, params):
        return history.select(
            'datetime', (pl.col('close') / pl.col('close').shift(4) - 1).alias('momentum'),
            (pl.col('datetime').dt.epoch('ns') + 900_000_000_000).alias('available_at_ns'),
        )

    train = recorded.head(200).with_columns(
        (pl.col('close') / pl.col('close').shift(4) - 1).alias('momentum'),
        (pl.col('close').shift(-1) / pl.col('open').shift(-1) - 1).alias('next_return'),
    ).drop_nulls(['momentum', 'next_return'])
    fitted = make_pipeline(StandardScaler(), Ridge()).fit(train.select('momentum'), train['next_return'])
    cutoff = datetime(2025, 1, 6, tzinfo=timezone.utc)
    python = Sensor.from_model(fitted, prepare=features, selection_cutoff=cutoff, output_id='next_bar_return')
    for options in ({}, {'selection_cutoff': cutoff}, {'map_output': lambda sensor, bar: (bar.prediction, None), 'mapped_output_id': 'target_exposure'}):
        with pytest.raises(ValueError):
            Cohort.from_python([native, python], **options)

    def allocation(sensor, bar):
        # Native class 0/1 means flat/long; a return forecast becomes signed size.
        value = bar.prediction if sensor is native else np.tanh(bar.prediction * 10000)
        return float(value), None

    mixed = Cohort.from_python(
        [native, python], map_output=allocation, mapped_output_id='target_exposure',
        selection_cutoff=cutoff,
    )
    raw = _after_training(recorded)
    native_rows, python_rows = native.predict_all(raw), python.predict_all(raw)
    observed = mixed.predict_all(raw)
    assert len(observed) == len(native_rows) == len(python_rows)
    valid = 0
    for result, compiled_row, python_row in zip(observed, native_rows, python_rows, strict=True):
        assert result.datetime == compiled_row.datetime == python_row.datetime
        assert result.available_at_ns is None
        assert result.trade_contract_digest is None
        if compiled_row.reason is not None or python_row.reason is not None:
            assert result.reason is not None
            assert result.prediction is result.probability is None
        else:
            expected = (compiled_row.prediction + np.tanh(python_row.prediction * 10000)) / 2
            assert result.reason is None and result.probability is None
            assert result.prediction == pytest.approx(expected, rel=1e-12, abs=1e-12)
            valid += 1
    assert valid
    assert mixed.predict(raw) == observed[-1]
    assert mixed.members == (native, python)
    assert mixed.origin == mixed.prediction_mode == 'python'
    assert mixed.manifest_id is mixed.cohort_id is None
    assert (native.manifest_id, native.permutation_id, native.trade_contract) == identity
    assert native.predict_all(raw) == native_rows


def _bound(selection, members):
    cohort = Cohort(rounds=selection)
    cohort.set_members(members)
    return cohort


def _assert_probability_oracle(cohort, members, raw):
    individual = [member.predict_all(raw) for member in members]
    actual = cohort.predict_all(raw)
    assert actual == cohort(raw)
    assert len(actual) == len(individual[0])
    for index, observed in enumerate(actual):
        row = [bars[index] for bars in individual]
        assert observed.datetime == row[0].datetime
        assert observed.trade_contract_digest is None
        assert observed.available_at_ns is None
        blocked = [bar for bar in row if bar.reason is not None]
        if blocked:
            assert observed.reason == blocked[0].reason
            assert observed.prediction is None
            assert observed.probability is None
        else:
            probability = sum(bar.probability for bar in row) / len(row)
            assert observed.reason is None
            assert observed.probability == pytest.approx(probability)
            assert observed.prediction == int(probability > 0.5)
    last = [member.predict(raw) for member in members]
    expected = sum(bar.probability for bar in last) / len(last)
    prediction = cohort.predict(raw)
    assert prediction.datetime == last[0].datetime
    assert prediction.reason is None
    assert prediction.probability == pytest.approx(expected)
    assert prediction.prediction == int(expected > 0.5)


def test_rounds_recorded_source_replay_and_aggregation(cohort_campaigns):
    campaigns, recorded = cohort_campaigns
    first, second, descendant = campaigns
    assert first.ids == second.ids
    assert first.sensors[0].manifest_id != second.sensors[0].manifest_id
    selected = [(first.directory, first.ids[0]), (first.directory, first.ids[1]), (second.directory, second.ids[0])]
    members = [first.sensors[0], first.sensors[1], second.sensors[0]]
    cohort = _bound(selected, list(reversed(members)))
    assert cohort.member_ids == [(member.manifest_id, member.permutation_id) for member in members]
    assert set(cohort.source_metadata) == {first.directory.resolve(), second.directory.resolve()}
    for attribute in ('experiment_dir', 'experiment_id', 'metadata', 'available_permutation_ids', 'permutation_ids', 'manifest_id'):
        assert getattr(cohort, attribute) is None
    raw = _after_training(recorded)
    _assert_probability_oracle(cohort, members, raw)
    # Votes use each selected member once, including unequal counts per source.
    cohort.aggregation_mode = 'majority_vote'
    votes = [member.predict_all(raw) for member in members]
    actual = cohort.predict_all(raw)
    assert actual == cohort(raw)
    for index, observed in enumerate(actual):
        row = [bars[index] for bars in votes]
        if all(bar.reason is None for bar in row):
            expected = sum(int(bar.prediction > 0.5) for bar in row) / len(row)
            assert observed.prediction == int(expected > 0.5)
            assert observed.probability is None
    last = [member.predict(raw) for member in members]
    prediction = cohort.predict(raw)
    expected = sum(int(bar.prediction > 0.5) for bar in last) / len(last)
    assert prediction.datetime == last[0].datetime
    assert prediction.reason is None
    assert prediction.prediction == int(expected > 0.5)
    assert prediction.probability is None
    narrowed_members = [first.sensors[0], descendant.sensors[0]]
    narrowed = _bound([(first.directory, first.ids[0]), (descendant.directory, descendant.ids[0])], narrowed_members)
    _assert_probability_oracle(narrowed, narrowed_members, raw)


def _old_hash(member, ids):
    payload = json.dumps({'aggregation_mode': 'probability_weighted',
                          'architecture_id': 'limen.sfd.reference_architecture.logreg_binary',
                          'manifest_id': member.manifest_id,
                          'permutation_ids': sorted(ids)}, sort_keys=True)
    return 'sha256:' + hashlib.sha256(payload.encode()).hexdigest()


def test_rounds_legacy_identity_and_relocation(cohort_campaigns, tmp_path):
    campaigns, recorded = cohort_campaigns
    first, second, _ = campaigns
    ids, members = first.ids[:2], first.sensors[:2]
    legacy = Cohort(experiment_log_path=str(first.directory), permutation_ids=ids)
    legacy.set_members(members)
    qualified = _bound([(str(first.directory), pid) for pid in ids], list(reversed(members)))
    assert legacy.cohort_id == qualified.cohort_id == _old_hash(members[0], ids)
    raw = _after_training(recorded)
    _assert_probability_oracle(legacy, members, raw)
    assert qualified(raw) == legacy(raw)
    assert qualified.predict(raw) == legacy.predict(raw)
    assert qualified.experiment_dir == first.directory.resolve()
    assert qualified.metadata == json.loads((first.directory / 'metadata.json').read_text())
    assert qualified.permutation_ids == ids
    copied = tmp_path / 'relocated'
    shutil.copytree(first.directory, copied)
    distributed = _bound([(first.directory, ids[0]), (copied, ids[1])], members)
    assert distributed.cohort_id == legacy.cohort_id
    assert distributed.manifest_id == members[0].manifest_id
    assert distributed.metadata is None
    selection = [(first.directory, ids[0]), (second.directory, ids[0])]
    mixed = _bound(selection, [members[0], second.sensors[0]])
    relocated = _bound([(second.directory, ids[0]), (copied, ids[0])], [second.sensors[0], members[0]])
    assert relocated.cohort_id == mixed.cohort_id
    payload = json.dumps({'architecture_id': mixed.architecture_id,
                          'aggregation_mode': mixed.aggregation_mode,
                          'rounds': sorted([[members[0].manifest_id, ids[0]],
                                           [second.sensors[0].manifest_id, ids[0]]])}, sort_keys=True)
    assert mixed.cohort_id == 'sha256:' + hashlib.sha256(payload.encode()).hexdigest()
    changed = _bound([(first.directory, ids[1]), selection[1]], [members[1], second.sensors[0]])
    assert changed.cohort_id != mixed.cohort_id


def _copy_campaign(campaign, destination):
    shutil.copytree(campaign.directory, destination)
    return destination


def _edit_metadata(directory, edit):
    path = directory / 'metadata.json'
    metadata = json.loads(path.read_text())
    edit(metadata)
    path.write_text(json.dumps(metadata))


def test_rounds_reject_invalid_sources_and_binding(cohort_campaigns, tmp_path):
    campaigns, _ = cohort_campaigns
    first, second, descendant = campaigns
    selection = [(first.directory, first.ids[0]), (second.directory, second.ids[0])]
    with pytest.raises(ValueError, match='non-empty'):
        Cohort(rounds=[])
    for legacy in ({'experiment_id': 'x'}, {'experiment_log_path': str(first.directory)},
                   {'permutation_ids': first.ids}, {'selector': 'all'}, {'selector_params': {}}):
        with pytest.raises(ValueError, match='rounds'):
            Cohort(rounds=selection, **legacy)
    alias = tmp_path / 'alias'
    alias.symlink_to(first.directory, target_is_directory=True)
    for path in (first.directory, alias):
        with pytest.raises(ValueError, match=r'duplicate|unique|repeated'):
            Cohort(rounds=[selection[0], (path, first.ids[0])])
    absent = next(pid for pid in first.ids if pid not in descendant.ids)
    with pytest.raises(ValueError, match=absent):
        Cohort(rounds=[(descendant.directory, absent)])
    missing_mid = _copy_campaign(first, tmp_path / 'missing_mid')
    _edit_metadata(missing_mid, lambda metadata: metadata.pop('manifest_id'))
    with pytest.raises(ValueError, match='manifest_id'):
        Cohort(rounds=[(missing_mid, first.ids[0])])
    cohort = Cohort(rounds=selection)
    for wrong_members in ([first.sensors[0]], [first.sensors[0], first.sensors[0]],
                          [first.sensors[0], descendant.sensors[0]]):
        with pytest.raises(ValueError):
            cohort.set_members(wrong_members)
        assert cohort.cohort_id is None
        with pytest.raises(RuntimeError, match='no bound members'):
            cohort.predict(pl.DataFrame())
    wrong_architecture = copy.copy(second.sensors[0])
    wrong_architecture._round_params = {**wrong_architecture.round_params, 'model_architecture': 'lightgbm_binary'}
    with pytest.raises(ValueError, match='architecture'):
        cohort.set_members([first.sensors[0], wrong_architecture])
    assert cohort.cohort_id is None


def test_rounds_reject_incompatible_sources_and_bars(cohort_campaigns, tmp_path):
    campaigns, _ = cohort_campaigns
    first, second, _ = campaigns
    source = _copy_campaign(second, tmp_path / 'source')
    def different_source(metadata):
        metadata['yaml_reference']['sfd']['manifest']['data_source']['params']['file_path_or_url'] = 'another-recorded-file.parquet'
    _edit_metadata(source, different_source)
    with pytest.raises(ValueError, match='source'):
        Cohort(rounds=[(first.directory, first.ids[0]), (source, second.ids[0])])
    architecture = _copy_campaign(second, tmp_path / 'architecture')
    _edit_metadata(architecture, lambda metadata: metadata['yaml_reference']['sfd']['manifest'].update(reference_architecture='limen.sfd.reference_architecture.lightgbm_binary'))
    changed = Cohort(rounds=[(first.directory, first.ids[0]), (architecture, second.ids[0])])
    with pytest.raises(ValueError, match='architecture'):
        changed.set_members([first.sensors[0], second.sensors[0]])
    walk_forward = _copy_campaign(second, tmp_path / 'walk_forward')
    _edit_metadata(walk_forward, lambda metadata: metadata['yaml_reference']['sfd']['manifest'].update(split_walk_forward={'n_folds': 2, 'test_bars': 24, 'purge_bars': 1, 'embargo_bars': 1, 'anchored': True}))
    with pytest.raises(ValueError, match=r'walk.forward|fold'):
        Cohort(rounds=[(walk_forward, second.ids[0])])
    bars = _copy_campaign(second, tmp_path / 'bars')
    _edit_metadata(bars, lambda metadata: metadata['yaml_reference']['sfd']['manifest'].update(bar_formation={'func': 'limen.data.bars.volume_bars', 'params': {'volume_threshold': '{C}'}}))
    records_path = _storage.round_path(bars)
    records = [json.loads(line) for line in _storage.text(records_path).splitlines()]
    for record in records:
        record['round_params']['bar_type'] = 'volume'
    records_path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0) for record in records))
    with pytest.raises(ValueError, match='bar'):
        Cohort(rounds=[(first.directory, first.ids[0]), (bars, second.ids[0])])
    # Even the same function declaration must compare the selected resolved arguments.
    other_bars = _copy_campaign(first, tmp_path / 'other_bars')
    _edit_metadata(other_bars, lambda metadata: metadata['yaml_reference']['sfd']['manifest'].update(bar_formation={'func': 'limen.data.bars.volume_bars', 'params': {'volume_threshold': '{C}'}}))
    other_records_path = _storage.round_path(other_bars)
    other_records = [json.loads(line) for line in _storage.text(other_records_path).splitlines()]
    for record in other_records:
        record['round_params']['bar_type'] = 'volume'
    other_records_path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0) for record in other_records))
    with pytest.raises(ValueError, match='bar'):
        Cohort(rounds=[(other_bars, first.ids[0]), (bars, second.ids[1])])


def test_rounds_reconstruction_failures_are_visible(cohort_campaigns, tmp_path):
    campaigns, recorded = cohort_campaigns
    first, second, _ = campaigns
    corrupted = _copy_campaign(second, tmp_path / 'corrupted')
    log = pl.read_csv(corrupted / 'results.csv')
    log.with_columns((pl.col('accuracy') + 1.0).alias('accuracy')).write_csv(corrupted / 'results.csv')
    cohort = Cohort(rounds=[(first.directory, first.ids[0]), (corrupted, second.ids[0])])
    with pytest.raises(ReconstructionError, match='metric mismatch'):
        Trainer(corrupted, data=recorded).train([second.ids[0]])
    assert cohort.cohort_id is None
    with pytest.raises(RuntimeError, match='no bound members'):
        cohort.predict(_after_training(recorded))


class _OutputFault:
    def __init__(self, sensor, transform):
        self.sensor = sensor
        self.transform = transform
        self.permutation_id = sensor.permutation_id
        self.manifest_id = sensor.manifest_id
        self.round_params = sensor.round_params
        self.prediction_mode = sensor.prediction_mode

    def predict(self, raw):
        return self.transform(self.sensor.predict(raw))

    def predict_all(self, raw):
        return [self.transform(bar) for bar in self.sensor.predict_all(raw)]


def test_rounds_alignment_and_inference_policies(cohort_campaigns):
    campaigns, recorded = cohort_campaigns
    first, second, descendant = campaigns
    selection = [(first.directory, first.ids[0]), (second.directory, second.ids[0])]
    members = [first.sensors[0], second.sensors[0]]
    raw = _after_training(recorded)
    def shifted(bar):
        return replace(bar, datetime=bar.datetime + timedelta(minutes=15) if bar.datetime is not None else None)
    for transform in (shifted, lambda bar: replace(bar, datetime=None) if bar.reason is None else bar):
        cohort = _bound(selection, [members[0], _OutputFault(members[1], transform)])
        for method in (cohort.predict, cohort.predict_all):
            with pytest.raises(ValueError, match=r'timestamp|datetime'):
                method(raw)
    cohort = _bound(selection, members)
    assert cohort.predict(raw.head(0)).reason == 'warm-up'
    assert cohort.predict(raw.head(1)).reason == 'warm-up'
    nulls = raw.with_columns(pl.when(pl.int_range(pl.len()) == 10).then(None).otherwise(pl.col('close')).alias('close'))
    assert cohort.predict_all(nulls)[10].reason == 'null-features'
    narrowed = _bound([(first.directory, first.ids[0]), (descendant.directory, descendant.ids[0])], [first.sensors[0], descendant.sensors[0]])
    guarded = recorded.filter(pl.col('datetime').dt.date().cast(pl.String) == '2025-01-06')
    assert first.sensors[0].predict(guarded).reason is None
    assert descendant.sensors[0].predict(guarded).reason == 'inside-training-window'
    assert narrowed.predict(guarded).reason == 'inside-training-window'
    healthy = members[0].predict_all(raw)
    with patch.object(members[1]._model, 'predict', side_effect=RuntimeError('recorded sensor failure')):
        assert cohort.predict(raw) == replace(members[0].predict(raw), trade_contract_digest=None, available_at_ns=None)
        assert cohort.predict_all(raw) == [replace(bar, trade_contract_digest=None, available_at_ns=None) for bar in healthy]
        with patch.object(members[0]._model, 'predict', side_effect=RuntimeError('recorded sensor failure')):
            assert cohort.predict(raw).reason == 'sensor-error'
            assert all(bar.reason == 'sensor-error' for bar in cohort.predict_all(raw))
    signed = _OutputFault(members[1], lambda bar: bar)
    signed.prediction_mode = 'target_exposure'
    with pytest.raises(ValueError, match=r'target.exposure'):
        Cohort(rounds=selection).set_members([members[0], signed])
    opposite = replace(members[1].predict(raw), probability=1.0 - members[0].predict(raw).probability)
    with patch.object(members[1], 'predict', return_value=opposite):
        assert cohort.predict(raw).probability == 0.5
        assert cohort.predict(raw).prediction == 0
    for invalid_probability in (float('nan'), -0.1, 1.1):
        invalid = _OutputFault(members[1], lambda bar, value=invalid_probability: replace(bar, probability=value) if bar.reason is None else bar)
        with pytest.raises(ValueError, match=r'finite|within'):
            _bound(selection, [members[0], invalid]).predict(raw)
