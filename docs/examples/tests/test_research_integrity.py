from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import signal
from unittest.mock import patch

import polars as pl
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from limen.sfd.reference_architecture import LogRegBinary
from limen.targets import QuantileBinaryTarget
from limen.backtest import trade_execution
from limen.backtest.execution_events import with_predictions

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures'


def files(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


def test_recorded_resume_refuses_changed_source_and_geometry(python_scenarios, compiled_scenarios, tmp_path, monkeypatch):
    wrong = tmp_path / 'wrong-source'
    wrong.mkdir()
    shutil.copyfile(FIXTURES / 'spot_15m_20250101_20250531.parquet', wrong / 'spot_1h_20240101_20241231.parquet')
    with pytest.raises(ValueError, match='Recorded input digest differs'):
        python_scenarios.hourly(wrong, 384)
    with pytest.raises(ValueError, match='Recorded input digest differs'):
        compiled_scenarios.configuration('recovery', wrong)

    config = compiled_scenarios.configuration('walk_forward', FIXTURES)
    config['sfd']['params']['C'] = [.5, 1.]
    config['uel']['n_permutations'] = 2
    bars = pl.read_parquet(FIXTURES / 'spot_1h_20240101_20241231.parquet', n_rows=384)
    directory = tmp_path / 'interrupted'

    def stop_after_one(log, msq):
        if log.height == 1:
            signal.raise_signal(signal.SIGINT)

    first = compiled_scenarios.execute(config, directory, bars=bars, callback=stop_after_one)
    assert first.experiment_log.height == 1
    before = files(directory)

    def forbidden(*args, **kwargs):
        raise AssertionError('Resume fitted a model before checking its recorded source and geometry')

    monkeypatch.setattr(LogRegBinary, 'train', forbidden)
    changed = deepcopy(config)
    changed['sfd']['manifest']['split_walk_forward']['embargo_bars'] += 1
    with pytest.raises(ValueError, match='split_walk_forward'):
        compiled_scenarios.execute(changed, directory, bars=bars, resume=True)
    assert files(directory) == before
    with pytest.raises(ValueError):
        compiled_scenarios.execute(config, directory, bars=bars.head(383), resume=True)
    assert files(directory) == before


def test_guarded_holdout_excludes_research_and_never_refits(compiled_scenarios, tmp_path, monkeypatch):
    from limen.experiment import _holdout_assessment as assessment
    from limen import Cohort
    cutoff = int(datetime(2024, 1, 5, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    trained, prefixes = [], []
    original_train = LogRegBinary.train
    original_execute = compiled_scenarios.execute
    original_freeze = compiled_scenarios.freeze_holdout_selection
    original_tape = assessment._tape

    def train(model, prepared, **params):
        provenance = prepared['holdout']
        assert provenance['source']['admitted']['rows'] == 96
        for bounds in provenance['splits'].values():
            if bounds['rows']:
                assert bounds['last_close_ns'] <= cutoff
        trained.append(deepcopy(provenance))
        return original_train(model, prepared, **params)

    def execute(*args, **kwargs):
        loop = original_execute(*args, **kwargs)
        assert loop.data.height == 96 and loop.data['datetime'].dt.epoch('ns').max() < cutoff
        return loop

    def forbidden(*args, **kwargs):
        raise AssertionError('Final assessment fitted a model or transform after selection was frozen')

    def freeze(cohort, **kwargs):
        frozen = original_freeze(cohort, **kwargs)
        for member in assessment.members(cohort):
            for value in member._fitted_params.values():
                if type(value).__module__.startswith(('limen.scalers.', 'limen.targets.')):
                    monkeypatch.setattr(type(value), '__init__', forbidden)
        monkeypatch.setattr(LogRegBinary, 'train', forbidden)
        monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
        # The assessment owns its frozen copy, even if the caller changes membership.
        cohort._members.clear()
        assert not cohort._members
        return frozen

    def tape(selection, raw, start, failures):
        assert isinstance(selection, Cohort) and len(assessment.members(selection)) == 1
        with ExitStack() as stack:
            for member in assessment.members(selection):
                original = member.predict

                def predict(frame, original=original):
                    clocks = frame['datetime'].dt.epoch('ns')
                    prefixes.append((frame.height, clocks[0], clocks[-1]))
                    return original(frame)

                stack.enter_context(patch.object(member, 'predict', side_effect=predict))
            return original_tape(selection, raw, start, failures)

    monkeypatch.setattr(LogRegBinary, 'train', train)
    monkeypatch.setattr(compiled_scenarios, 'execute', execute)
    monkeypatch.setattr(compiled_scenarios, 'freeze_holdout_selection', freeze)
    monkeypatch.setattr(assessment, '_tape', tape)
    result = compiled_scenarios.holdout(FIXTURES, tmp_path / 'holdout')
    assert result['rounds'] == 1 and len(trained) >= 2
    assert [row[0] for row in prefixes] == list(range(25, 49))
    assert all(row[1] == cutoff-24*3_600_000_000_000 for row in prefixes)
    assert [row[2] for row in prefixes] == [cutoff+index*3_600_000_000_000 for index in range(24)]


def test_fold_fitting_calibration_and_objective_are_local(compiled_scenarios, tmp_path, monkeypatch):
    config = compiled_scenarios.configuration('walk_forward', FIXTURES)
    bars = pl.read_parquet(FIXTURES / 'spot_1h_20240101_20241231.parquet', n_rows=384)
    trained, calibrated, targets = [], [], []
    original_train, original_calibration, original_target = LogRegBinary.train, CalibratedClassifierCV.fit, QuantileBinaryTarget.__init__

    def train(model, prepared, **params):
        trained.append((model, prepared))
        return original_train(model, prepared, **params)

    def calibrate(model, x, y, **params):
        calibrated.append((model, x, y))
        return original_calibration(model, x, y, **params)

    def target(model, train_data, *args, **params):
        targets.append((model, train_data))
        original_target(model, train_data, *args, **params)

    monkeypatch.setattr(LogRegBinary, 'train', train)
    monkeypatch.setattr(CalibratedClassifierCV, 'fit', calibrate)
    monkeypatch.setattr(QuantileBinaryTarget, '__init__', target)
    directory = tmp_path / 'folds'
    loop = compiled_scenarios.execute(config, directory, bars=bars)
    assert len(trained) == len(calibrated) == len(targets) == 2
    assert trained[0][0] is not trained[1][0] and calibrated[0][0] is not calibrated[1][0]
    assert trained[0][1]['_scaler'] is not trained[1][1]['_scaler'] and targets[0][0] is not targets[1][0]
    with gzip.open(directory / 'round_data.jsonl.gz', 'rt') as stream:
        record = json.loads(next(stream))
    for fold, ((model, prepared), (calibrator, x, y), (label, source), nested) in enumerate(zip(trained, calibrated, targets, record['folds'], strict=True)):
        test_start = 288 + fold*48
        pool_end = test_start-4
        validation_start = pool_end*8//9
        raw_fit, raw_val, raw_test = bars.head(validation_start-4), bars[validation_start:pool_end], bars[test_start:test_start+48]
        assert source['datetime'].to_list() == raw_fit['datetime'].to_list()
        close = raw_fit['close'].to_numpy()
        changes = np.sort((close[1:]/close[:-1]-1)*100)
        assert label.cutoff == pytest.approx(changes[len(changes)//2], rel=1e-12, abs=1e-12)
        context = prepared['_trade_context']
        for rows, raw in zip(context.model_rows, (raw_fit, raw_val, raw_test), strict=True):
            assert rows.height > 0 and rows['datetime'].is_in(raw['datetime'].implode()).all()
        assert x.equals(prepared['x_val']) and y.equals(prepared['y_val'])
        assert calibrator.estimator.estimator is model.model
        expected = model.predict({'x_test': prepared['x_test']})
        np.testing.assert_array_equal(nested['preds'], expected['_preds'])
        row = loop.fold_results.filter(pl.col('fold') == fold).row(0, named=True)
        probabilities = model.predict({'x_test': prepared['x_val']})['_probs']
        candidates = [2., *np.unique(np.minimum(np.arange(.2, .7+.05, .05), .7))[::-1].tolist()]
        scores = [trade_execution(with_predictions(context.partitions[1], (probabilities >= threshold).astype(float)), context.policy).metrics['total_return'] for threshold in candidates]
        best = max(range(len(scores)), key=lambda index: scores[index])
        assert row['optimal_threshold'] == pytest.approx(candidates[best])
        assert row['val_backtest_total_return'] == pytest.approx(scores[best], rel=1e-12, abs=1e-12)
        assert row['val_score'] == row['val_backtest_total_return']
