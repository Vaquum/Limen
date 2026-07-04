import json
import math
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest.mock import patch

import polars as pl
import pytest

from limen.cli.commands.run import run_experiment
from limen.cohort import Cohort
from limen.cohort.sfc.top_n import select as select_top_n
from limen.data import historical_data
from limen.inference import ReconstructionError
from limen.inference import Trainer
from limen.inference.sensor import BarPrediction


_E2E_LOGREG_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: test_logreg
      mode: development
    sfd:
      manifest:
        type: ml
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 3600
        split_dates:
          train_start: "2025-01-01"
          train_end: "2025-01-15"
          val_start: "2025-01-15"
          val_end: "2025-01-18"
          test_start: "2025-01-18"
          test_end: "2025-01-22"
        indicators:
          - func: limen.indicators.roc
            params:
              period: 1
              group: momentum
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
        C: [1.0, 0.5]
        random_state: [42]
    uel:
      n_permutations: 2
      search_strategy:
        type: grid
      output_format: csv
''')


_E2E_RANDOM_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: test_random
      mode: development
    sfd:
      manifest:
        type: ml
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 3600
        split_dates:
          train_start: "2025-01-01"
          train_end: "2025-01-15"
          val_start: "2025-01-15"
          val_end: "2025-01-18"
          test_start: "2025-01-18"
          test_end: "2025-01-22"
        indicators:
          - func: limen.indicators.roc
            params:
              period: 1
              group: momentum
        target:
          name: quantile_flag
          class: limen.targets.QuantileBinaryTarget
          fit_params:
            source_column: roc_1
            quantile: 0.5
          transform_params:
            shift: -1
        reference_architecture: limen.sfd.reference_architecture.random_binary
      params:
        random_weights: [0.5]
    uel:
      n_permutations: 1
      search_strategy:
        type: grid
      output_format: csv
''')


_E2E_ABLATION_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: test_ablation
      mode: development
    sfd:
      manifest:
        type: ml
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 3600
        split_dates:
          train_start: "2025-01-01"
          train_end: "2025-01-15"
          val_start: "2025-01-15"
          val_end: "2025-01-18"
          test_start: "2025-01-18"
          test_end: "2025-01-22"
        indicators:
          - func: limen.indicators.roc
            params:
              period: 1
              group: momentum
          - func: limen.indicators.roc
            params:
              period: 2
              group: momentum
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
        feature_ablation:
          drop_count_key: feature_drop_count
          seed_key: feature_drop_seed
        reference_architecture: limen.sfd.reference_architecture.logreg_binary
      params:
        scaler_type: [logreg]
        C: [1.0]
        random_state: [42]
        feature_drop_count: [1]
        feature_drop_seed: [42]
    uel:
      n_permutations: 1
      search_strategy:
        type: grid
      output_format: csv
''')


_E2E_REPRO_ROLLING_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: test_repro_rolling
      mode: development
    sfd:
      manifest:
        type: ml
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 3600
        split_dates:
          train_start: "2025-01-01"
          train_end: "2025-01-15"
          val_start: "2025-01-15"
          val_end: "2025-01-18"
          test_start: "2025-01-18"
          test_end: "2025-01-22"
          val_predict_guard: false
          test_predict_guard: false
        indicators:
          - func: limen.indicators.roc
            params:
              period: 1
              group: momentum
        target:
          name: quantile_flag
          class: limen.targets.QuantileBinaryTarget
          fit_params:
            source_column: roc_1
            quantile: 0.5
          transform_params:
            shift: -1
        scaler:
          class: limen.scalers.CausalRollingRobustScaler
          params:
            window: scaler_window
            min_samples: 5
        strict_mode: true
        reference_architecture: limen.sfd.reference_architecture.logreg_binary
      params:
        C: [1.0]
        random_state: [42]
        scaler_window: [20, 50]
    uel:
      n_permutations: 2
      search_strategy:
        type: grid
      output_format: csv
''')


_E2E_REPRO_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: test_repro
      mode: development
    sfd:
      manifest:
        type: ml
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 3600
        split_dates:
          train_start: "2025-01-01"
          train_end: "2025-01-15"
          val_start: "2025-01-15"
          val_end: "2025-01-18"
          test_start: "2025-01-18"
          test_end: "2025-01-22"
          val_predict_guard: false
          test_predict_guard: false
        indicators:
          - func: limen.indicators.roc
            params:
              period: 1
              group: momentum
        target:
          name: quantile_flag
          class: limen.targets.QuantileBinaryTarget
          fit_params:
            source_column: roc_1
            quantile: 0.5
          transform_params:
            shift: -1
        reference_architecture: limen.sfd.reference_architecture.logreg_binary
      params:
        C: [1.0]
        random_state: [42]
    uel:
      n_permutations: 1
      search_strategy:
        type: grid
      output_format: csv
''')


def _make_e2e_data(kline_size: int = 3600,
                   n_rows: int | None = None,
                   start_date_limit: object = None,
                   end_date_limit: object = None) -> pl.DataFrame:
    _ = kline_size, start_date_limit, end_date_limit
    n = int(n_rows or 500)
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n)]
    close = [100.0 + 0.02 * i + math.sin(i / 7.0) for i in range(n)]
    return pl.DataFrame({
        'datetime': timestamps,
        'open': [v - 0.1 for v in close],
        'high': [v + 0.2 for v in close],
        'low': [v - 0.2 for v in close],
        'close': close,
        'volume': [1000.0 + i for i in range(n)],
    })


def _run_e2e_experiment(experiment_dir: Path, yaml_text: str) -> list[int]:
    experiment_dir = Path(experiment_dir).resolve()
    original = historical_data.HistoricalData.get_spot_klines
    historical_data.HistoricalData.get_spot_klines = staticmethod(_make_e2e_data)
    try:
        with TemporaryDirectory() as tmpdir, patch('click.echo'), patch('click.secho'):
            yaml_path = Path(tmpdir) / 'exp.yaml'
            yaml_path.write_text(yaml_text.replace(
                'output_format: csv',
                f'output_format: csv\n  output_path: "{experiment_dir}"',
            ))
            run_experiment(yaml_path)
    finally:
        historical_data.HistoricalData.get_spot_klines = original
    round_ids: list[int] = []
    with (experiment_dir / 'round_data.jsonl').open('r') as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            round_ids.append(json.loads(stripped)['round_id'])
    return sorted(round_ids)


def _train_e2e(experiment_dir: Path, pids: list[int]) -> tuple[Trainer, list]:
    original = historical_data.HistoricalData.get_spot_klines
    historical_data.HistoricalData.get_spot_klines = staticmethod(_make_e2e_data)
    try:
        trainer = Trainer(experiment_dir)
        sensors = trainer.train(pids)
    finally:
        historical_data.HistoricalData.get_spot_klines = original
    return trainer, sensors


def test_trainer_requires_yaml_reference() -> None:

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'metadata.json').write_text(
            json.dumps({'sfd_module': 'limen.sfd.foundational_sfd.logreg_binary'}),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        with pytest.raises(ValueError, match='yaml_reference'):
            Trainer(experiment_dir, data=pl.DataFrame({'x': [1, 2, 3]}))


def test_trainer_rejects_non_dict_yaml_reference() -> None:

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'metadata.json').write_text(
            json.dumps({'yaml_reference': []}),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        with pytest.raises(ValueError, match='yaml_reference'):
            Trainer(experiment_dir, data=pl.DataFrame({'x': [1, 2, 3]}))


def test_trainer_rejects_malformed_yaml_reference() -> None:

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'metadata.json').write_text(
            json.dumps({'yaml_reference': {'metadata': {'name': 'yaml_exp'}, 'sfd': {}}}),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        with pytest.raises(ValueError, match='yaml_reference'):
            Trainer(experiment_dir, data=pl.DataFrame({'x': [1, 2, 3]}))


def test_load_round_data_skips_blank_lines_and_loads_valid_entries() -> None:
    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'round_data.jsonl').write_text(
            '\n'
            '{"round_id": 0, "round_params": {"alpha": 1}}\n'
            '{"round_id": 1, "round_params": {"alpha": 2}}\n'
        )

        trainer = object.__new__(Trainer)
        trainer._experiment_dir = experiment_dir

        round_data = trainer._load_round_data()

    assert list(round_data) == ['0', '1']
    assert round_data['1']['round_params'] == {'alpha': 2}


def test_load_round_data_rejects_malformed_jsonl() -> None:
    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'round_data.jsonl').write_text(
            '{"round_id": 0, "round_params": {}}\n'
            'not-json\n'
        )

        trainer = object.__new__(Trainer)
        trainer._experiment_dir = experiment_dir

        with pytest.raises(ValueError, match=r'Malformed JSON in round_data\.jsonl line 2'):
            trainer._load_round_data()


def test_load_round_data_rejects_missing_required_fields() -> None:
    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'round_data.jsonl').write_text(
            '{"round_id": 0}\n'
        )

        trainer = object.__new__(Trainer)
        trainer._experiment_dir = experiment_dir

        with pytest.raises(ValueError, match='requires round_id and round_params'):
            trainer._load_round_data()


def test_load_round_data_rejects_non_object_round_params() -> None:
    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'round_data.jsonl').write_text(
            '{"round_id": 0, "round_params": []}\n'
        )

        trainer = object.__new__(Trainer)
        trainer._experiment_dir = experiment_dir

        with pytest.raises(ValueError, match='round_params must be an object'):
            trainer._load_round_data()


def test_load_round_data_requires_round_data_file() -> None:
    with TemporaryDirectory() as tmpdir:
        trainer = object.__new__(Trainer)
        trainer._experiment_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError, match=r'round_data\.jsonl'):
            trainer._load_round_data()


def test_load_original_log_returns_none_when_results_csv_is_missing() -> None:
    with TemporaryDirectory() as tmpdir:
        trainer = object.__new__(Trainer)
        trainer._experiment_dir = Path(tmpdir)

        assert trainer._load_original_log() is None


def test_validate_metrics_ignores_metadata_fields_and_accepts_small_stochastic_drift() -> None:
    trainer = object.__new__(Trainer)
    trainer._original_log = {
        7: {
            'id': 7,
            'param_alpha': 1.0,
            'accuracy': 0.8000001,
            'precision': 0.2000001,
            'label': 'baseline',
        }
    }
    trainer._param_keys = frozenset({'param_alpha'})

    mismatches = trainer._validate_metrics(
        7,
        {
            'id': 7,
            'param_alpha': 99.0,
            'accuracy': 0.8000002,
            'precision': 0.2000002,
            '_probs': [0.1, 0.9],
            'label': 'changed-string-is-ignored',
        },
        is_deterministic=False,
    )

    assert mismatches == []


def test_validate_metrics_reports_missing_permutations_and_large_deterministic_mismatches() -> None:
    trainer = object.__new__(Trainer)
    trainer._original_log = {
        3: {
            'accuracy': 0.80,
            'precision': 0.20,
        }
    }
    trainer._param_keys = frozenset()

    assert trainer._validate_metrics(99, {'accuracy': 0.5}, True) == [
        'permutation 99 not found in results.csv'
    ]

    mismatches = trainer._validate_metrics(
        3,
        {'accuracy': 0.81, 'precision': 0.20},
        is_deterministic=True,
    )

    assert mismatches == ['accuracy: original=0.8, new=0.81']


def test_trainer_yaml_end_to_end() -> None:
    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_ids = _run_e2e_experiment(exp_dir, _E2E_LOGREG_YAML)
        assert len(round_ids) == 2

        metadata = json.loads((exp_dir / 'metadata.json').read_text())
        assert 'yaml_reference' in metadata
        assert 'limen_version' in metadata
        assert 'manifest_id' in metadata
        assert metadata['manifest_id'].startswith('sha256:')

        trainer, sensors = _train_e2e(exp_dir, round_ids)
        assert len(sensors) == 2

        expected_mid = metadata['manifest_id']
        for i, sensor in enumerate(sensors):
            assert sensor.permutation_id == round_ids[i]
            assert isinstance(sensor.round_params, dict)
            assert sensor.manifest_id == expected_mid

        with pytest.raises(ValueError, match='not found in round_data'):
            trainer.train([999])


def test_trainer_yaml_reconstruction_error_stochastic() -> None:
    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_ids = _run_e2e_experiment(exp_dir, _E2E_RANDOM_YAML)
        assert len(round_ids) == 1

        original = historical_data.HistoricalData.get_spot_klines
        historical_data.HistoricalData.get_spot_klines = staticmethod(_make_e2e_data)
        try:
            trainer = Trainer(exp_dir)
            with pytest.raises(ReconstructionError, match='metric mismatch'):
                trainer.train(round_ids)
        finally:
            historical_data.HistoricalData.get_spot_klines = original


def test_trainer_yaml_reconstruction_error_tampered_log() -> None:
    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_ids = _run_e2e_experiment(exp_dir, _E2E_LOGREG_YAML)

        df = pl.read_csv(exp_dir / 'results.csv')
        df = df.with_columns((pl.col('accuracy') + 0.5).alias('accuracy'))
        df.write_csv(exp_dir / 'results.csv')

        with pytest.raises(ReconstructionError, match='metric mismatch'):
            _train_e2e(exp_dir, [round_ids[0]])


def test_trainer_yaml_logreg_stochastic_validation() -> None:
    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_ids = _run_e2e_experiment(exp_dir, _E2E_LOGREG_YAML)
        _, sensors = _train_e2e(exp_dir, [round_ids[0]])
        assert len(sensors) == 1
        assert sensors[0]._model.deterministic is False


def test_trainer_yaml_sensor_inference() -> None:
    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_ids = _run_e2e_experiment(exp_dir, _E2E_LOGREG_YAML)
        trainer, sensors = _train_e2e(exp_dir, [round_ids[0]])
        sensor = sensors[0]

        # underlying model can still be called directly
        data_dict = trainer._manifest.prepare_data(trainer._data, sensor.round_params)
        x_batch = data_dict['x_test'].to_numpy()[:2]
        model_result = sensor._model.predict({'x_test': x_batch})
        assert isinstance(model_result, dict)
        assert '_preds' in model_result

        # DataFrame path — returns BarPrediction for last bar
        n = 100
        post_ts = [datetime(2025, 1, 22) + timedelta(hours=i) for i in range(n)]
        close = [110.0 + 0.02 * i + math.sin(i / 7.0) for i in range(n)]
        post_data = pl.DataFrame({
            'datetime': post_ts,
            'open': [v - 0.1 for v in close],
            'high': [v + 0.2 for v in close],
            'low': [v - 0.2 for v in close],
            'close': close,
            'volume': [1000.0 + i for i in range(n)],
        })
        bar_pred = sensor.predict(post_data)
        assert isinstance(bar_pred, BarPrediction)
        assert bar_pred.reason is None
        assert bar_pred.prediction is not None


def test_trainer_yaml_feature_ablation() -> None:
    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_ids = _run_e2e_experiment(exp_dir, _E2E_ABLATION_YAML)
        assert len(round_ids) == 1
        _, sensors = _train_e2e(exp_dir, round_ids)
        sensor = sensors[0]
        dropped = sensor.round_params.get('_dropped_features')
        assert isinstance(dropped, list)
        assert len(dropped) == 1


def test_sensor_reproduces_training_metrics_on_val_test() -> None:
    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_ids = _run_e2e_experiment(exp_dir, _E2E_REPRO_YAML)
        assert len(round_ids) == 1

        training_accuracy = (
            pl.read_csv(exp_dir / 'results.csv')
            .filter(pl.col('id') == str(round_ids[0]))['accuracy'][0]
        )

        trainer, sensors = _train_e2e(exp_dir, round_ids)
        sensor = sensors[0]

        bar_preds = sensor.predict_all(trainer._data)

        test_start = datetime(2025, 1, 18)
        test_end = datetime(2025, 1, 22)
        test_preds = sorted(
            [p for p in bar_preds if p.datetime is not None and p.reason is None
             and test_start <= p.datetime < test_end],
            key=lambda p: p.datetime,
        )

        data_dict = trainer._manifest.prepare_data(trainer._data, sensor.round_params)
        y_test = data_dict['y_test'].to_list()

        # sensor predicts for all test bars including the last one (no target); y_test drops it
        test_preds = test_preds[:len(y_test)]

        assert len(test_preds) == len(y_test), (
            f'expected {len(y_test)} test predictions, got {len(test_preds)}'
        )

        n_correct = sum(1 for p, y in zip(test_preds, y_test, strict=False) if p.prediction == y)
        sensor_accuracy = n_correct / len(y_test)

        # results.csv stores metrics rounded to 3 decimal places
        assert round(sensor_accuracy, 3) == training_accuracy, (
            f'sensor accuracy {sensor_accuracy} does not match training accuracy {training_accuracy}'
        )


def test_sensor_reproduces_training_metrics_with_rolling_scaler() -> None:
    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        exp_dir.mkdir()
        round_ids = _run_e2e_experiment(exp_dir, _E2E_REPRO_ROLLING_YAML)
        assert len(round_ids) == 2

        results = pl.read_csv(exp_dir / 'results.csv')

        assert 'scaler_window' in results.columns
        assert set(results['scaler_window'].to_list()) == {20, 50}

        # strict_mode=true: clean synthetic data must never trigger StrictModeError
        assert 'strict_mode_error' in results.columns
        assert results['strict_mode_error'].null_count() == len(round_ids), (
            'clean data must produce no strict_mode_error on any round'
        )

        assert results['accuracy'].null_count() == 0
        assert results['auc'].null_count() == 0

        for row in results.iter_rows(named=True):
            assert row['scaler_window'] in {20, 50}, f'scaler_window={row["scaler_window"]} is a jumbled value'
            assert isinstance(row['accuracy'], float) and 0.0 <= row['accuracy'] <= 1.0, (
                f'accuracy={row["accuracy"]} outside [0,1] — possible column jumble'
            )
            assert isinstance(row['auc'], float) and 0.0 <= row['auc'] <= 1.0, (
                f'auc={row["auc"]} outside [0,1] — possible column jumble'
            )
            assert row['C'] == 1.0, f'C={row["C"]} does not match the only swept value'
            assert row['random_state'] == 42, f'random_state={row["random_state"]} does not match the only swept value'
            assert row['strict_mode_error'] is None, f'strict_mode_error unexpectedly set: {row["strict_mode_error"]}'

        cohort = Cohort(
            experiment_log_path=str(exp_dir),
            selector=select_top_n,
            selector_params={'column': 'accuracy', 'n': 2},
        )
        assert len(cohort.permutation_ids) == 2

        trainer, sensors = _train_e2e(exp_dir, round_ids)
        assert len(sensors) == 2

        assert {s.round_params['scaler_window'] for s in sensors} == {20, 50}

        val_start = datetime(2025, 1, 15)
        test_start = datetime(2025, 1, 18)
        test_end = datetime(2025, 1, 22)

        raw = trainer._data
        train_raw = raw.filter(pl.col('datetime') < val_start)
        val_raw = raw.filter((pl.col('datetime') >= val_start) & (pl.col('datetime') < test_start))
        test_raw = raw.filter((pl.col('datetime') >= test_start) & (pl.col('datetime') < test_end))

        for sensor in sensors:
            w = sensor.round_params['scaler_window']

            data_dict = trainer._manifest.prepare_data(trainer._data, sensor.round_params)
            y_val = data_dict['y_val'].to_list()
            y_test = data_dict['y_test'].to_list()

            # Feed cco+split per split: sensor's rolling computation operates on the same-shaped
            # array as prepare_data, giving FP-identical scaled features.
            # n_raw_cco = cco_indicator_rows + scaler_context_rows; roc(period=1) gives 1 leading null.
            N_raw_cco = w + 1
            val_preds_raw = sensor.predict_all(pl.concat([train_raw.tail(N_raw_cco), val_raw]))
            test_preds_raw = sensor.predict_all(pl.concat([val_raw.tail(N_raw_cco), test_raw]))

            # CCO bars are excluded by the datetime filter; no reason filter needed.
            val_preds = sorted(
                [p for p in val_preds_raw if p.datetime is not None and val_start <= p.datetime < test_start],
                key=lambda p: p.datetime,
            )
            test_preds = sorted(
                [p for p in test_preds_raw if p.datetime is not None and test_start <= p.datetime < test_end],
                key=lambda p: p.datetime,
            )

            # CCO must recover all val and test rows regardless of window size.
            # Without CCO, the first (min_samples - 1) = 4 val rows would be
            # excluded as cold-scaler warm-up, giving len(val_preds) < len(y_val) + 1.
            assert len(val_preds) == len(y_val) + 1, (
                f'CCO failed on val (window={w}): '
                f'expected {len(y_val) + 1}, got {len(val_preds)}'
            )
            assert len(test_preds) == len(y_test) + 1, (
                f'CCO failed on test (window={w}): '
                f'expected {len(y_test) + 1}, got {len(test_preds)}'
            )

            n_correct = sum(
                1 for p, y in zip(test_preds[:len(y_test)], y_test, strict=False)
                if p.prediction == y
            )
            sensor_accuracy = n_correct / len(y_test)
            training_accuracy = (
                results.filter(pl.col('id') == str(sensor.permutation_id))['accuracy'][0]
            )
            assert round(sensor_accuracy, 3) == training_accuracy, (
                f'sensor (window={w}) accuracy {sensor_accuracy:.4f} '
                f'!= training {training_accuracy}'
            )

        cohort.set_members(sensors)

        cohort_all = cohort.predict_all(trainer._data)
        assert len(cohort_all) == len(trainer._data)
        valid_cohort = [p for p in cohort_all if p.reason is None]
        assert len(valid_cohort) > 0
        assert all(p.prediction in (0, 1) for p in valid_cohort)
        assert all(p.probability is not None and 0.0 <= p.probability <= 1.0 for p in valid_cohort)

        cohort_last = cohort.predict(trainer._data)
        assert isinstance(cohort_last, BarPrediction)
        assert cohort_last.prediction in (0, 1)
        assert cohort_last.probability is not None and 0.0 <= cohort_last.probability <= 1.0
        last_valid_cohort = next(p for p in reversed(cohort_all) if p.reason is None)
        assert cohort_last.prediction == last_valid_cohort.prediction
