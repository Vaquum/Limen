"""Short recorded-data warning and runtime comparison at sklearn boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'
CASES = ('legacy_polars', 'explicit_arrays')
LIMITS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'POLARS_MAX_THREADS')
INTERCHANGE_WARNING = 'Support for the dataframe interchange protocol is deprecated since version 1.40.0'


def _manifest(binary):
    from limen.data import HistoricalData
    from limen.experiment import MLManifest
    from limen.features import lag_range
    from limen.indicators import window_return
    from limen.scalers import RobustScaler
    from limen.sfd.reference_architecture import logreg_binary, ridge_regressor
    from limen.targets import NextReturnTarget, QuantileBinaryTarget

    manifest = (MLManifest()
                .set_data_source(HistoricalData.get_any_file,
                                 params={'file_path_or_url': str(FIXTURE)})
                .set_split_dates(datetime(2024, 1, 1), datetime(2024, 4, 1),
                                 datetime(2024, 4, 1), datetime(2024, 4, 8),
                                 datetime(2024, 4, 8), datetime(2024, 4, 15))
                .add_indicator(window_return, period=1)
                .add_feature(lag_range, col='ret_1', start=1, end=13))
    if binary:
        manifest.with_target_label('up_next', QuantileBinaryTarget,
                                   fit_params={'source_column': 'ret_1', 'quantile': 0.5},
                                   transform_params={'shift': -1})
    else:
        manifest.with_target_label('next_return', NextReturnTarget,
                                   transform_params={'periods': 1, 'scale': 100.0})
    return (manifest.set_scaler(RobustScaler).set_strict_mode(True)
            .with_reference_architecture(logreg_binary if binary else ridge_regressor)
            .set_backtest_config(fee_bps=5.0, slip_bps=2.0, notional_rate=1.0))


def _configuration(mode):
    from limen.calibration import grid_threshold_optimizer, sklearn_probability_calibrator
    from limen.experiment.manifest_core import CalibrationConfig

    if mode == 'none':
        return None
    return CalibrationConfig(
        calibration_func=None if mode == 'threshold_only' else sklearn_probability_calibrator,
        calibration_params={} if mode == 'threshold_only' else {'method': mode},
        threshold_func=grid_threshold_optimizer,
        threshold_params={'threshold_min': 0.3, 'threshold_max': 0.7, 'threshold_step': 0.1},
    )


def _logreg(case, data, params, mode):
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression

    from limen.calibration import grid_threshold_optimizer
    from limen.sfd.reference_architecture import LogRegBinary

    config = _configuration(mode)
    if case == 'explicit_arrays':
        model = LogRegBinary(prediction_calibration_config=config).train(data, **params)
        return model, model.predict(data)
    model = LogRegBinary()
    model.model = LogisticRegression(**params).fit(data['x_train'], data['y_train'])
    if config is None:
        return model, {'_preds': model.model.predict(data['x_test']),
                       '_probs': model.model.predict_proba(data['x_test'])[:, 1]}
    fitted = model.model
    if mode != 'threshold_only':
        fitted = CalibratedClassifierCV(FrozenEstimator(fitted), method=mode).fit(
            data['x_val'], data['y_val'])
    threshold, score = grid_threshold_optimizer(
        data['y_val'], fitted.predict_proba(data['x_val'])[:, 1], **config.threshold_params)
    probs = fitted.predict_proba(data['x_test'])[:, 1]
    return model, {'_preds': (probs >= threshold).astype(np.int8), '_probs': probs,
                   'optimal_threshold': threshold, 'val_score': score}


def _ridge(case, data, params):
    from sklearn.linear_model import Ridge
    from limen.sfd.reference_architecture import RidgeRegressor

    model = RidgeRegressor()
    if case == 'explicit_arrays':
        model.train(data, **params)
        result = model.predict(data)
    else:
        model.model = Ridge(**params).fit(data['x_train'], data['y_train'])
        result = {'_preds': model.model.predict(data['x_test'])}
    return model, result


def _outcome(model, data, result, binary):
    import numpy as np
    from limen.metrics.binary_metrics import binary_metrics
    from limen.metrics.continuous_metrics import continuous_metrics

    preds = result['_preds']
    direction = preds if binary else (preds > 0).astype(int)
    actual = data['y_test'] if binary else (np.asarray(data['y_test']) > 0).astype(int)
    metrics = (binary_metrics(data, preds, result['_probs']) if binary
               else continuous_metrics(data, preds))
    metrics.update(model._compute_confusion(direction, actual, data['price_data_for_backtest']))
    metrics.update(model._compute_backtest(direction, data))
    for name in ('optimal_threshold', 'val_score'):
        metrics[name] = result.get(name)
    # Preserve undefined metrics explicitly in standards-compliant JSON.
    metrics = {name: (str(float(value)) if isinstance(value, (float, np.floating))
                     and not np.isfinite(value) else value) for name, value in metrics.items()}
    return {'predictions': np.asarray(preds).tolist(),
            'probabilities': np.asarray(result['_probs']).tolist() if binary else None,
            'metrics': metrics, 'feature_names': data['x_train'].columns,
            'split_rows': {split: data[f'x_{split}'].height for split in ('train', 'val', 'test')}}


def _campaign(case, output):
    import numpy as np
    import polars as pl
    import sklearn
    from threadpoolctl import threadpool_info, threadpool_limits
    from limen import __version__

    stages = {'preparation_seconds': 0.0, 'model_seconds': 0.0, 'metrics_seconds': 0.0}
    outcomes = {}
    started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        recorded = pl.read_parquet(FIXTURE)
        for mode in ('none', 'threshold_only', 'isotonic', 'sigmoid', 'ridge'):
            for parameter in (0.25, 1.0):
                binary = mode != 'ridge'
                begin = time.perf_counter()
                data = _manifest(binary).prepare_data(recorded, {})
                stages['preparation_seconds'] += time.perf_counter() - begin
                params = ({'C': parameter, 'solver': 'lbfgs', 'tol': 1e-8,
                           'max_iter': 200, 'random_state': 42} if binary
                          else {'alpha': parameter, 'solver': 'cholesky', 'copy_X': True,
                                'fit_intercept': True, 'random_state': 42})
                begin = time.perf_counter()
                model, result = (_logreg(case, data, params, mode) if binary
                                 else _ridge(case, data, params))
                stages['model_seconds'] += time.perf_counter() - begin
                begin = time.perf_counter()
                outcomes[f'{mode}/{parameter}'] = _outcome(model, data, result, binary)
                outcomes[f'{mode}/{parameter}']['parameters'] = params
                stages['metrics_seconds'] += time.perf_counter() - begin
        pools = threadpool_info()
        if pl.thread_pool_size() != 1 or any(pool['num_threads'] != 1 for pool in pools):
            raise AssertionError(f'Expected one native thread: {pools}')
    observations = Counter((warning.category.__name__, str(warning.message)) for warning in caught)
    warning_records = [{'category': category, 'message': message, 'count': count}
                       for (category, message), count in sorted(observations.items())]
    exact_count = sum(count for (_, message), count in observations.items()
                      if message == INTERCHANGE_WARNING)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {'case': case, 'limen_version': __version__, 'python': sys.version,
              'platform': platform.platform(), 'versions': {'numpy': np.__version__,
              'polars': pl.__version__, 'sklearn': sklearn.__version__},
              'rounds': len(outcomes), 'outcomes': outcomes, 'stages': stages,
              'campaign_seconds': time.perf_counter() - started,
              'exact_interchange_warning_count': exact_count, 'warnings': warning_records,
              'native_pools': pools,
              'peak_process_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024)}
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    sys.stdout.write(json.dumps({key: report[key] for key in
                                ('case', 'rounds', 'stages', 'warnings')}) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.case is not None:
        _campaign(args.case, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    started = time.perf_counter()
    for case in CASES:
        path = args.output / f'{case}.json'
        begin = time.perf_counter()
        subprocess.run([sys.executable, '-m', 'tests.benchmark_sklearn_input_boundaries',
                        '--case', case, '--output', str(path)], cwd=ROOT,
                       env={**os.environ, **dict.fromkeys(LIMITS, '1')}, check=True, timeout=60)
        reports[case] = json.loads(path.read_text())
        reports[case]['end_to_end_seconds'] = time.perf_counter() - begin
    import numpy as np
    baseline, explicit = (reports[case] for case in CASES)
    for key, before in baseline['outcomes'].items():
        after = explicit['outcomes'][key]
        for field in ('feature_names', 'split_rows', 'parameters'):
            if before[field] != after[field]:
                raise AssertionError(f'{key}/{field} changed')
        for field in ('predictions', 'probabilities'):
            if before[field] is not None:
                np.testing.assert_allclose(after[field], before[field], rtol=1e-9, atol=1e-12)
        if before['metrics'] != after['metrics']:
            raise AssertionError(f'{key} metrics changed: {before["metrics"]} != {after["metrics"]}')
    if explicit['exact_interchange_warning_count']:
        raise AssertionError('Explicit boundary still invokes deprecated dataframe interchange')
    report = {'fixture': str(FIXTURE), 'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
              'baseline': 'Pre-change raw Polars sklearn fit, calibration fit and validation/test prediction; same installed versions and recorded observations',
              'source_base_commit': '04bc9411fde2c6fd64277212d21ae91d7d71e97e',
              'thread_limits': dict.fromkeys(LIMITS, '1'),
              'split_dates': {'train': ['2024-01-01', '2024-04-01'],
                              'validation': ['2024-04-01', '2024-04-08'],
                              'test': ['2024-04-08', '2024-04-15']},
              'measurement': 'Fresh-process import, recorded-data loading, per-round preparation, fit/calibration/prediction, existing metric helpers and report writes; no execution-loop throughput claim',
              'interchange_warning_exact_message': INTERCHANGE_WARNING, 'campaigns': reports,
              'outputs_equal': True, 'prediction_rtol': 1e-9, 'prediction_atol': 1e-12,
              'end_to_end_ratio': baseline['end_to_end_seconds'] / explicit['end_to_end_seconds'],
              'total_elapsed_seconds': time.perf_counter() - started}
    destination = args.output / 'benchmark.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    sys.stdout.write(json.dumps({'report': str(destination), 'outputs_equal': True,
                                'end_to_end_ratio': report['end_to_end_ratio'],
                                'exact_warning_counts': {case: reports[case]['exact_interchange_warning_count']
                                                         for case in CASES}}) + '\n')


if __name__ == '__main__':
    main()
