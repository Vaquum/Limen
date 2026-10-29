"""Check recorded adaptive constituents and stopping boundaries independently."""
import hashlib
import gzip
import json
from datetime import datetime

import numpy as np
import polars as pl
import pytest


def verify_adaptive(result, output, inputs):
    points = pl.read_parquet(inputs / 'recorded-trade-points.parquet')
    assert result['trade_rows'] == points.height == 817
    assert result['expected_num_ticks'] == 20 and result['ema_span'] == 10
    assert {row['kind'] for row in result['formed']} == {'imbalance', 'run'}
    signed = points.select((pl.col('close').diff().sign().replace(0, None).forward_fill().fill_null(0)
                            * pl.col('volume')).alias('signed'))['signed']
    positive, negative = signed.clip(lower_bound=0), (-signed).clip(lower_bound=0)
    imbalance_mean = np.abs(signed.ewm_mean(span=10, adjust=False).to_numpy())
    run_mean = np.maximum(positive.ewm_mean(span=10, adjust=False).to_numpy(),
                          negative.ewm_mean(span=10, adjust=False).to_numpy())
    for row in result['formed']:
        path = output / (row['kind'] + '.parquet')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        formed = pl.read_parquet(path)
        assert formed.height == row['rows'] > 2
        mean = run_mean if row['kind'] == 'run' else imbalance_mean
        cursor, expected_length = 0, 20.
        for bar in formed.iter_rows(named=True):
            length = bar['bar_count']
            end = cursor + length
            source = points.slice(cursor, length)
            assert bar['datetime'] == source['datetime'][0] and bar['base_interval'] == 0
            assert bar['open'] == source['open'][0] and bar['close'] == source['close'][-1]
            assert bar['high'] == source['high'].max() and bar['low'] == source['low'].min()
            for column in ('volume', 'liquidity_sum', 'maker_volume', 'maker_liquidity'):
                assert bar[column] == pytest.approx(source[column].sum(), rel=1e-12)
            evidence = (np.maximum(positive.slice(cursor, length).cum_sum(), negative.slice(cursor, length).cum_sum())
                        if row['kind'] == 'run' else np.abs(signed.slice(cursor, length).cum_sum()))
            evidence = np.asarray(evidence)
            threshold = expected_length * (mean[cursor:end] if cursor == 0 else mean[cursor - 1])
            crosses = (np.arange(cursor, end) + 1 >= 20) & (evidence > 0) & (evidence >= threshold - 1e-12)
            assert np.flatnonzero(crosses).tolist() == [length - 1]
            expected_length += (2 / 11) * (length - expected_length)
            cursor = end
        assert cursor == row['consumed_trade_rows']
        assert row['omitted_trade_rows'] == points.height - cursor
        assert row['last_available_at_ns'] == points['available_at_ns'][cursor - 1]
        assert row['volume'] == pytest.approx(points['volume'].head(cursor).sum(), rel=1e-12)


def verify_adaptive_cli(project):
    from limen.data.utils import compute_data_bars
    directory = project / 'results/dev/compiled-bars-adaptive'
    log = pl.read_csv(directory / 'results.csv')
    assert log.height == log['id'].n_unique() == 2
    if 'strict_mode_error' in log.columns:
        assert log['strict_mode_error'].drop_nulls().len() == 0
    with gzip.open(directory / 'round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 2 and {row['round_params']['bar_type'] for row in records} == {'imbalance', 'run'}
    points = pl.read_parquet(project / 'inputs/recorded-trade-points.parquet')
    rows = []
    for record in records:
        kind = record['round_params']['bar_type']
        assert len(record['folds']) == 2
        errors = []
        for fold in record['folds']:
            index = fold['fold']
            source = points.slice(points.height - 450 + index * 225, 225)
            formed = compute_data_bars(source, bar_type=kind, expected_num_ticks=5, ema_span=10)
            labeled = formed.with_columns(((pl.col('close').shift(-1) / pl.col('close') - 1) * 100).alias('label'))
            dates = [datetime.fromisoformat(value) for value in fold['alignment']['test_datetimes']]
            truth = labeled.filter(pl.col('datetime').is_in(dates))['label'].to_numpy()
            predictions = np.asarray(fold['preds'])
            assert predictions.size == truth.size > 0
            assert np.isfinite(predictions).all() and np.isfinite(truth).all()
            rmse = float(np.sqrt(np.mean((predictions - truth) ** 2)))
            assert fold['results']['rmse'] == pytest.approx(rmse, rel=1e-10, abs=1e-12)
            promotion = fold['fold_promotion']['unavailable_reason']['code']
            assert promotion == 'clock'
            rows.append({'kind': kind, 'fold': index, 'raw_test_rows': source.height,
                         'formed_test_rows': formed.height, 'predictions': predictions.size,
                         'rmse': rmse, 'promotion_unavailable_reason': promotion})
            errors.append(rmse)
        assert log.filter(pl.col('id') == record['round_id'])['rmse'][0] == pytest.approx(np.mean(errors))
    return {'rounds': log.height, 'source_rows': points.height, 'folds': rows}
