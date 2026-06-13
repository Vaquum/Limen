import logging
import math
from datetime import date
from datetime import datetime
from datetime import timedelta

import polars as pl
import pytest

from limen.experiment import MLManifest
from limen.experiment.errors import StrictModeError
from limen.indicators import roc
from limen.scalers.causal_rolling_robust_scaler import CausalRollingRobustScaler
from limen.scalers.robust_scaler import RobustScaler
from limen.targets import QuantileBinaryTarget
from limen.yaml.compiler import build_manifest
from limen.yaml.parser import parse


_WINDOW = 20
_MIN_SAMPLES = 5
_ROC_PERIOD = 1

# 300 hourly bars: 2025-01-01 00:00 to 2025-01-13 11:00
# train: [2025-01-01, 2025-01-09) = 192 bars
# val:   [2025-01-09, 2025-01-11) = 48 bars
# test:  [2025-01-11, 2025-01-13) = 48 bars
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

    # sensor_input_prep applies indicators + scaler continuously on the full raw series
    fitted_params = data_dict['_fitted_params']
    sensor_data, _ = manifest.sensor_input_prep(raw, fitted_params, {})

    # filter sensor output to val timestamps; this is the ground-truth continuous output
    val_dt_start = datetime(_VAL_START.year, _VAL_START.month, _VAL_START.day)
    val_dt_end = datetime(_VAL_END.year, _VAL_END.month, _VAL_END.day)
    val_continuous = (
        sensor_data
        .filter((pl.col('datetime') >= val_dt_start) & (pl.col('datetime') < val_dt_end))
        .drop('datetime')
    )

    feature_cols = data_dict['x_val'].columns
    val_cco = data_dict['x_val']  # 47 rows (last target row dropped)

    # compare first 47 rows (sensor includes all 48 val rows; x_val drops last due to target shift)
    assert len(val_cco) == len(val_continuous) - 1
    for col in feature_cols:
        cco_vals = val_cco[col].to_list()
        cont_vals = val_continuous[col].to_list()[:len(val_cco)]
        for i, (a, b) in enumerate(zip(cco_vals, cont_vals)):
            assert abs(a - b) < 1e-9, f"val col={col} row={i}: CCO={a} continuous={b}"


def test_cco_row_count_recovery() -> None:
    raw = _make_raw_data()

    # with rolling scaler: CCO recovers indicator warm-up rows (1 for roc period=1)
    data_dict_cco = _prepare(_make_manifest(use_rolling_scaler=True), raw)

    # with stateless scaler: no CCO context rows beyond indicator warm-up
    data_dict_stateless = _prepare(_make_manifest(use_rolling_scaler=False), raw)

    # CCO should recover at least as many rows as stateless (warm-up rows are not dropped)
    assert len(data_dict_cco['x_val']) >= len(data_dict_stateless['x_val'])
    assert len(data_dict_cco['x_test']) >= len(data_dict_stateless['x_test'])


def test_no_cco_for_stateless_scaler() -> None:
    raw = _make_raw_data()
    data_dict = _prepare(_make_manifest(use_rolling_scaler=False), raw)

    # stateless scaler has context_rows=0; scaler values are globally fitted so no CCO needed
    # verify alignment is still correct and row counts are within window bounds
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


def test_non_strict_mode_warns_on_gap(caplog: pytest.LogCaptureFixture) -> None:
    raw = _make_raw_data()

    rows = raw.to_dicts()
    rows[200]['close'] = None
    raw_with_gap = pl.DataFrame(rows)

    manifest = _make_manifest(strict_mode=False)
    with caplog.at_level(logging.WARNING, logger='limen.experiment.manifest_core'):
        data_dict = _prepare(manifest, raw_with_gap)

    assert any('Unexpected nulls' in r.message for r in caplog.records)
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
