import csv as _csv
import io
import logging
import math
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

import limen.sfd.foundational_sfd.random_binary as _random_binary_sfd
from limen.experiment import MLManifest
from limen.experiment.errors import StrictModeError
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.indicators import roc
from limen.scalers.causal_rolling_robust_scaler import CausalRollingRobustScaler
from limen.scalers.robust_scaler import RobustScaler
from limen.targets import QuantileBinaryTarget
from limen.yaml.compiler import build_manifest
from limen.yaml.parser import parse


_WINDOW = 20
_MIN_SAMPLES = 5
_ROC_PERIOD = 1

_N = 300
_TRAIN_START = date(2025, 1, 1)
_TRAIN_END = date(2025, 1, 9)
_VAL_START = date(2025, 1, 9)
_VAL_END = date(2025, 1, 11)
_TEST_START = date(2025, 1, 11)
_TEST_END = date(2025, 1, 13)
_VAL_HOURS = 48
_TEST_HOURS = 48


def _make_raw_data(n: int = _N) -> pl.DataFrame:
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n)]
    close = [100.0 + 0.05 * i + math.sin(i / 8.0) for i in range(n)]
    return pl.DataFrame({
        'datetime': timestamps,
        'open': [v - 0.1 for v in close],
        'high': [v + 0.2 for v in close],
        'low': [v - 0.2 for v in close],
        'close': close,
        'volume': [1000.0 + i for i in range(n)],
    })


def _make_manifest(use_rolling_scaler: bool = True, strict_mode: bool = False) -> MLManifest:
    m = (
        MLManifest()
        .set_split_dates(_TRAIN_START, _TRAIN_END, _VAL_START, _VAL_END, _TEST_START, _TEST_END)
        .add_indicator(roc, period=_ROC_PERIOD, group='momentum')
        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'roc_1', 'quantile': 0.5},
            transform_params={'shift': -1},
        )
    )
    if use_rolling_scaler:
        m.set_scaler(CausalRollingRobustScaler, extra_params={'window': _WINDOW, 'min_samples': _MIN_SAMPLES})
    else:
        m.set_scaler(RobustScaler)
    if strict_mode:
        m.set_strict_mode(True)
    return m


def _prepare(manifest: MLManifest, raw: pl.DataFrame) -> dict:
    return manifest.prepare_data(raw, round_params={})


def test_context_rows_property() -> None:
    train = pl.DataFrame({'x': [float(i) for i in range(50)]})
    scaler = CausalRollingRobustScaler(train, window=500, min_samples=10)
    assert scaler.context_rows == 500


def test_context_rows_default_window() -> None:
    train = pl.DataFrame({'x': [float(i) for i in range(100)]})
    scaler = CausalRollingRobustScaler(train)
    assert scaler.context_rows == 1000


def test_cco_no_contamination() -> None:
    raw = _make_raw_data()
    data_dict = _prepare(_make_manifest(), raw)

    # datetime is stripped from x_val/x_test; verify via alignment and row-count bounds
    assert len(data_dict['x_val']) <= _VAL_HOURS, 'val has more rows than val window — contamination'
    assert len(data_dict['x_test']) <= _TEST_HOURS, 'test has more rows than test window — contamination'

    alignment = data_dict['_alignment']
    first_test_dt = alignment['first_test_datetime']
    last_test_dt = alignment['last_test_datetime']
    first_test_date = first_test_dt.date() if hasattr(first_test_dt, 'date') else first_test_dt
    last_test_date = last_test_dt.date() if hasattr(last_test_dt, 'date') else last_test_dt
    assert first_test_date >= _TEST_START, f'first test row {first_test_date} is before test_start'
    assert last_test_date < _TEST_END, f'last test row {last_test_date} is on or after test_end'


def test_cco_val_matches_continuous() -> None:
    raw = _make_raw_data()
    manifest = _make_manifest()
    data_dict = _prepare(manifest, raw)

    fitted_params = data_dict['_fitted_params']
    sensor_data, _ = manifest.sensor_input_prep(raw, fitted_params, {})

    val_dt_start = datetime(_VAL_START.year, _VAL_START.month, _VAL_START.day)
    val_dt_end = datetime(_VAL_END.year, _VAL_END.month, _VAL_END.day)
    val_continuous = (
        sensor_data
        .filter((pl.col('datetime') >= val_dt_start) & (pl.col('datetime') < val_dt_end))
        .drop('datetime')
    )

    feature_cols = data_dict['x_val'].columns
    val_cco = data_dict['x_val']

    # compare first 47 rows (sensor includes all 48 val rows; x_val drops last due to target shift)
    assert len(val_cco) == len(val_continuous) - 1
    for col in feature_cols:
        cco_vals = val_cco[col].to_list()
        cont_vals = val_continuous[col].to_list()[:len(val_cco)]
        for i, (a, b) in enumerate(zip(cco_vals, cont_vals, strict=False)):
            assert abs(a - b) < 1e-9, f"val col={col} row={i}: CCO={a} continuous={b}"


def test_cco_row_count_recovery() -> None:
    raw = _make_raw_data()

    data_dict_cco = _prepare(_make_manifest(use_rolling_scaler=True), raw)
    data_dict_stateless = _prepare(_make_manifest(use_rolling_scaler=False), raw)

    assert len(data_dict_cco['x_val']) >= len(data_dict_stateless['x_val'])
    assert len(data_dict_cco['x_test']) >= len(data_dict_stateless['x_test'])


def test_no_cco_for_stateless_scaler() -> None:
    raw = _make_raw_data()
    data_dict = _prepare(_make_manifest(use_rolling_scaler=False), raw)

    # stateless scaler has context_rows=0; scaler values are globally fitted so no CCO needed
    assert len(data_dict['x_val']) <= _VAL_HOURS
    assert len(data_dict['x_test']) <= _TEST_HOURS

    alignment = data_dict['_alignment']
    first_test_dt = alignment['first_test_datetime']
    first_test_date = first_test_dt.date() if hasattr(first_test_dt, 'date') else first_test_dt
    assert first_test_date >= _TEST_START


def test_strict_mode_raises_on_gap() -> None:
    raw = _make_raw_data()

    rows = raw.to_dicts()
    rows[200]['close'] = None
    raw_with_gap = pl.DataFrame(rows)

    manifest = _make_manifest(strict_mode=True)
    with pytest.raises(StrictModeError, match='Unexpected nulls'):
        _prepare(manifest, raw_with_gap)


def test_non_strict_mode_warns_on_gap() -> None:
    raw = _make_raw_data()

    rows = raw.to_dicts()
    rows[200]['close'] = None
    raw_with_gap = pl.DataFrame(rows)

    manifest = _make_manifest(strict_mode=False)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger('limen.experiment.manifest_core')
    logger.addHandler(handler)
    try:
        data_dict = _prepare(manifest, raw_with_gap)
    finally:
        logger.removeHandler(handler)

    assert 'Unexpected nulls' in buf.getvalue()
    assert 'x_val' in data_dict


def test_scaler_params_yaml_window() -> None:
    from textwrap import dedent

    yaml_text = dedent('''\
        schema_version: "1.0"
        metadata:
          name: test_scaler_params
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
              train_end: "2025-01-09"
              val_start: "2025-01-09"
              val_end: "2025-01-11"
              test_start: "2025-01-11"
              test_end: "2025-01-13"
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
                window: 50
                min_samples: 10
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

    yaml_dict, parse_errors = parse(yaml_text)
    assert not parse_errors, parse_errors

    manifest = build_manifest(yaml_dict)
    raw = _make_raw_data()
    data_dict = manifest.prepare_data(raw, round_params={'C': 1.0, 'random_state': 42})

    fitted_scaler = data_dict['_fitted_params'].get('_scaler')
    assert fitted_scaler is not None
    assert fitted_scaler.window == 50
    assert fitted_scaler.context_rows == 50


def test_scaler_params_default_when_omitted() -> None:
    manifest = _make_manifest()
    raw = _make_raw_data()
    data_dict = manifest.prepare_data(raw, round_params={})

    fitted_scaler = data_dict['_fitted_params'].get('_scaler')
    assert fitted_scaler is not None
    assert fitted_scaler.window == _WINDOW
    assert fitted_scaler.context_rows == _WINDOW


def _make_dynamic_scaler_yaml(scaler_block: str,
                               extra_params_block: str = '',
                               scaler_window: int = 60) -> str:
    from textwrap import dedent
    return dedent(f'''\
        schema_version: "1.0"
        metadata:
          name: test_dynamic_scaler
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
              train_end: "2025-01-09"
              val_start: "2025-01-09"
              val_end: "2025-01-11"
              test_start: "2025-01-11"
              test_end: "2025-01-13"
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
            {scaler_block}
            reference_architecture: limen.sfd.reference_architecture.logreg_binary
          params:
            C: [1.0]
            random_state: [42]
            scaler_window: [{scaler_window}]
            {extra_params_block}
        uel:
          n_permutations: 1
          search_strategy:
            type: grid
          output_format: csv
    ''')


def test_scaler_params_class_form_with_dynamic_window() -> None:
    scaler_block = '''\
scaler:
              class: limen.scalers.CausalRollingRobustScaler
              params:
                window: scaler_window'''

    yaml_dict, parse_errors = parse(_make_dynamic_scaler_yaml(scaler_block, scaler_window=60))
    assert not parse_errors, parse_errors

    manifest = build_manifest(yaml_dict)
    raw = _make_raw_data()
    data_dict = manifest.prepare_data(raw, round_params={'C': 1.0, 'random_state': 42, 'scaler_window': 60})

    fitted_scaler = data_dict['_fitted_params'].get('_scaler')
    assert fitted_scaler is not None
    assert fitted_scaler.window == 60, f'expected window=60, got {fitted_scaler.window}'
    assert fitted_scaler.context_rows == 60


def test_scaler_params_from_params_with_dynamic_window() -> None:
    scaler_block = '''\
scaler:
              from_params: scaler_type
              params:
                window: scaler_window'''
    extra_params_block = 'scaler_type: [causal_rolling_robust]'

    yaml_dict, parse_errors = parse(_make_dynamic_scaler_yaml(scaler_block, extra_params_block, scaler_window=60))
    assert not parse_errors, parse_errors

    manifest = build_manifest(yaml_dict)
    raw = _make_raw_data()
    data_dict = manifest.prepare_data(
        raw,
        round_params={'C': 1.0, 'random_state': 42, 'scaler_window': 60, 'scaler_type': 'causal_rolling_robust'},
    )

    fitted_scaler = data_dict['_fitted_params'].get('_scaler')
    assert fitted_scaler is not None
    assert fitted_scaler.window == 60, f'expected window=60, got {fitted_scaler.window}'
    assert fitted_scaler.context_rows == 60


def test_scaler_params_mixed_static_and_dynamic() -> None:
    # window is a sweep param reference (dynamic); min_samples is a literal (static)
    scaler_block = '''\
scaler:
              class: limen.scalers.CausalRollingRobustScaler
              params:
                window: scaler_window
                min_samples: 3'''

    yaml_dict, parse_errors = parse(_make_dynamic_scaler_yaml(scaler_block, scaler_window=30))
    assert not parse_errors, parse_errors

    manifest = build_manifest(yaml_dict)
    raw = _make_raw_data()
    data_dict = manifest.prepare_data(raw, round_params={'C': 1.0, 'random_state': 42, 'scaler_window': 30})

    fitted_scaler = data_dict['_fitted_params'].get('_scaler')
    assert fitted_scaler is not None
    assert fitted_scaler.window == 30, f'expected window=30 (dynamic), got {fitted_scaler.window}'
    assert fitted_scaler.min_samples == 3, f'expected min_samples=3 (static), got {fitted_scaler.min_samples}'


def test_uel_continues_after_strict_mode_error() -> None:
    domain = ParamDomain(_random_binary_sfd.params())
    strategy = RandomStrategy(domain, seed=42)
    uel = UniversalExperimentLoop(sfd=_random_binary_sfd, search_strategy=strategy, test_mode=True)

    call_count = [0]
    original_prep = uel.prep

    def patched_prep(data: pl.DataFrame, round_params: dict | None = None) -> dict:
        call_count[0] += 1
        if call_count[0] == 2:
            raise StrictModeError('Unexpected nulls in val split · Checkpoint A · roc_1 @ [2025-01-09T08:00]')
        return original_prep(data, round_params)

    uel.prep = patched_prep

    with TemporaryDirectory() as tmpdir:
        uel._run_with_msq(
            experiment_name=str(Path(tmpdir) / 'test'),
            n_permutations=3,
            context_params=None,
            resume=False,
            post_processing=True,
        )

    assert uel.experiment_log is not None
    assert uel.experiment_log.shape[0] == 3, 'all 3 rounds must appear in experiment log'
    assert len(uel.round_params) == 2, 'only successful rounds in round_params'
    assert len(uel.preds) == 2, 'only successful rounds in preds'
    assert len(uel._alignment) == 2, 'only successful rounds in alignment'
    assert 'strict_mode_error' in uel.experiment_log.columns, 'strict_mode_error column must exist'
    error_rows = uel.experiment_log.filter(pl.col('strict_mode_error').is_not_null())
    ok_rows = uel.experiment_log.filter(pl.col('strict_mode_error').is_null())
    assert error_rows.shape[0] == 1, 'exactly one round must have strict_mode_error'
    assert ok_rows.shape[0] == 2, 'successful rounds must have null strict_mode_error'
    assert 'Unexpected nulls' in error_rows['strict_mode_error'][0]
    # successful rounds must have metric columns populated (not dropped due to header mismatch)
    assert 'id' in ok_rows.columns and ok_rows['id'].null_count() == 0


_UNDER_WARM_WINDOW = 25
_SHORT_N = 80
_SHORT_TRAIN_START = date(2025, 1, 1)
_SHORT_TRAIN_END = date(2025, 1, 2)    # 24 train rows < n_raw_cco=26 → under-warming
_SHORT_VAL_START = date(2025, 1, 2)
_SHORT_VAL_END = date(2025, 1, 3)
_SHORT_TEST_START = date(2025, 1, 3)
_SHORT_TEST_END = date(2025, 1, 4)


def _make_short_raw_data() -> pl.DataFrame:
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(_SHORT_N)]
    close = [100.0 + 0.05 * i for i in range(_SHORT_N)]
    return pl.DataFrame({
        'datetime': timestamps,
        'open': [v - 0.1 for v in close],
        'high': [v + 0.2 for v in close],
        'low': [v - 0.2 for v in close],
        'close': close,
        'volume': [1000.0 + float(i) for i in range(_SHORT_N)],
    })


def _make_under_warm_manifest(strict_mode: bool = False) -> MLManifest:
    m = (
        MLManifest()
        .set_split_dates(
            _SHORT_TRAIN_START, _SHORT_TRAIN_END,
            _SHORT_VAL_START, _SHORT_VAL_END,
            _SHORT_TEST_START, _SHORT_TEST_END,
        )
        .add_indicator(roc, period=1, group='momentum')
        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'roc_1', 'quantile': 0.5},
            transform_params={'shift': -1},
        )
        .set_scaler(CausalRollingRobustScaler, extra_params={'window': _UNDER_WARM_WINDOW, 'min_samples': 3})
    )
    if strict_mode:
        m.set_strict_mode(True)
    return m


def test_cco_under_warm_strict_mode_raises() -> None:
    raw = _make_short_raw_data()
    manifest = _make_under_warm_manifest(strict_mode=True)
    with pytest.raises(StrictModeError, match='under-warmed CCO'):
        manifest.prepare_data(raw, round_params={})


def test_cco_under_warm_non_strict_warns() -> None:
    raw = _make_short_raw_data()
    manifest = _make_under_warm_manifest(strict_mode=False)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger('limen.experiment.manifest_core')
    logger.addHandler(handler)
    try:
        data_dict = manifest.prepare_data(raw, round_params={})
    finally:
        logger.removeHandler(handler)
    assert 'under-warmed CCO' in buf.getvalue()
    assert 'x_val' in data_dict


def test_uel_first_round_strict_mode_error_does_not_corrupt_csv_header() -> None:
    domain = ParamDomain(_random_binary_sfd.params())
    strategy = RandomStrategy(domain, seed=42)
    uel = UniversalExperimentLoop(sfd=_random_binary_sfd, search_strategy=strategy, test_mode=True)

    call_count = [0]
    original_prep = uel.prep

    def patched_prep(data: pl.DataFrame, round_params: dict | None = None) -> dict:
        call_count[0] += 1
        if call_count[0] == 1:
            raise StrictModeError('Unexpected nulls in val split · Checkpoint A · roc_1 @ [2025-01-01T00:00]')
        return original_prep(data, round_params)

    uel.prep = patched_prep

    with TemporaryDirectory() as tmpdir:
        exp_name = str(Path(tmpdir) / 'test_header')
        uel._run_with_msq(
            experiment_name=exp_name,
            n_permutations=3,
            context_params=None,
            resume=False,
            post_processing=True,
        )
        csv_path = Path(f'{exp_name}.csv')
        assert csv_path.exists(), 'CSV must be written'
        with csv_path.open('r', newline='') as f:
            reader = _csv.reader(f)
            header = next(reader)
            rows = list(reader)

    assert 'accuracy' in header, f'accuracy missing from CSV header — first-round failure corrupted the schema: {header}'
    assert len(rows) == 3, f'expected 3 rows in CSV, got {len(rows)}'
    acc_idx = header.index('accuracy')
    filled = [r[acc_idx] for r in rows if r[acc_idx] != '']
    assert len(filled) == 2, 'exactly 2 successful rounds should have accuracy values in CSV'
