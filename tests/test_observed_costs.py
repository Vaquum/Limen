from __future__ import annotations

import copy
import hashlib
import json
import socket
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from limen.backtest import _cost_identity, _cost_statistics
from limen.backtest.observed_costs import CostCalibration

FIXTURE = Path(__file__).parent / 'fixtures/observed_costs'
CUTOFF = int(datetime(2026, 8, 4, 2, 33, 11, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
GOLDENS = {
    'buy': ('0.08705296739230516936814656738', '1.676299555845261135650477426', 245, 142),
    'sell': ('0.1098781198244829665254772317', '4.748190846357268517885019166', 118, 53),
}


def _history(tmp_path, member='admitted.json'):
    provenance = json.loads((FIXTURE / 'provenance.json').read_text())
    with zipfile.ZipFile(FIXTURE / 'history.zip') as archive:
        content = archive.read(member)
    assert hashlib.sha256(content).hexdigest() == provenance['members'][member]['sha256']
    path = tmp_path / member
    path.write_bytes(content)
    return path, json.loads(content)


def _write_fault(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')))
    return path


def _no_network(*args, **kwargs):
    raise AssertionError('Recorded-cost API attempted network access')


def _no_estimation(*args, **kwargs):
    raise AssertionError('Pinned artifact read attempted re-estimation')


def test_recorded_parent_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, 'socket', _no_network)
    path, facts = _history(tmp_path)
    calibration = CostCalibration.from_history(path, cutoff_ns=CUTOFF)
    report = calibration.report()
    assert set(report) == {'digest', 'method', 'source', 'clocks', 'estimates', 'support', 'sensitivity', 'exclusions', 'limitations'}
    assert report['method'] == 'minute_midpoint_1s_v1'
    assert report['clocks']['available_at_ns'] == CUTOFF
    assert report['clocks']['historical_delivery'] == 'unknown'
    assert report['clocks']['created_at_ns'] >= report['clocks']['available_at_ns']
    assert report['support']['selected_legs'] == report['support']['admitted_legs'] == 363
    assert report['support']['disposition'] == 'retrospective_low_effective_support'
    assert report['support']['parent_completeness'] == 'requested_and_terminal_unknown'
    for side, (price, fee, legs, parents) in GOLDENS.items():
        actual = report['estimates'][side]
        assert abs(Decimal(actual['price_cost_bps']) - Decimal(price)) <= Decimal('1e-12')
        assert abs(Decimal(actual['fee_bps']) - Decimal(fee)) <= Decimal('1e-12')
        assert (actual['legs'], actual['parents'], actual['days']) == (legs, parents, 3)
        assert actual['fee_transfer'] == 'unavailable_account_specific'
        assert len(report['sensitivity']['computed_selected_population'][side]['leave_one_day_out']) == 3
    assert Decimal(report['estimates']['buy']['parent_effective_n']) == pytest.approx(Decimal('22.20500128601606857178213607'))
    assert Decimal(report['estimates']['sell']['parent_effective_n']) == pytest.approx(Decimal('2.305309405301448476604581045'))
    assert report['support']['low_effective_support_sides'] == ['sell']
    assert report['estimates']['buy']['builder_fee_present_legs'] + report['estimates']['sell']['builder_fee_present_legs'] == 46
    assert report['estimates']['buy']['twap_legs'] == 10
    assert report['estimates']['sell']['twap_legs'] == 0
    expected_slip = (Decimal(GOLDENS['buy'][0]) + Decimal(GOLDENS['sell'][0])) / 2
    assert Decimal(report['estimates']['slip_bps']) == expected_slip

    first = facts['fills'][0]
    frame = pl.DataFrame({'time_ns': [first['time_ns']],
                          'price': [float(Decimal(first['quote_notional']) / Decimal(first['quantity']))],
                          'side': [first['side']], 'is_taker': [first['role'] == 'taker']})
    assert hashlib.sha256(_cost_identity.canonical(frame)).hexdigest() == '470195e5391c6c2b30e16fec67f3ae5487e04c9e0bc7f61af40dce5d7ba482ee'
    assert _cost_identity.digest((first['time_ns'],)) != _cost_identity.digest([first['time_ns']])
    _assert_canonical_identity(frame)
    artifact = tmp_path / 'calibration.json'
    expected_digest = calibration.digest  # Persist separately from the artifact in the caller specification.
    assert calibration.write(artifact) == calibration.write(artifact) == expected_digest
    with monkeypatch.context() as context:
        context.setattr(_cost_statistics, 'report', _no_estimation)
        restored = CostCalibration.read(artifact, expected_digest=expected_digest)
    assert restored.report() == calibration.report()
    report['estimates'].clear()
    assert calibration.report()['estimates']['buy']['parents'] == 142
    with pytest.raises(AttributeError):
        calibration._payload = {}

    fault = json.loads(artifact.read_text())
    fault['payload']['report']['estimates']['buy']['parents'] += 1
    corrupted = _write_fault(tmp_path / 'fault-artifact-content.json', fault)
    with pytest.raises(ValueError, match='pinned digest'):
        CostCalibration.read(corrupted, expected_digest=expected_digest)
    with pytest.raises(FileExistsError):
        calibration.write(corrupted)
    whitespace_fault = tmp_path / 'fault-artifact-whitespace.json'
    whitespace_fault.write_bytes(artifact.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='canonical JSON'):
        CostCalibration.read(whitespace_fault, expected_digest=expected_digest)
    with pytest.raises(ValueError, match='pinned digest'):
        CostCalibration.read(artifact, expected_digest='0' * 64)

    for field, expected in [('reference_price', 'unavailable'), ('reference_received_at_ns', 'unavailable'), ('fee', 'retrospective_low_effective_support')]:
        fault = copy.deepcopy(facts)
        fault['fills'][0][field] = None
        changed = CostCalibration.from_history(_write_fault(tmp_path / f'fault-missing-{field}.json', fault), cutoff_ns=CUTOFF).report()
        assert changed['support']['disposition'] == expected
        if field == 'fee':
            assert changed['estimates']['buy']['fee_bps'] is None
            assert changed['estimates']['slip_bps'] == calibration.report()['estimates']['slip_bps']
    # Identity and currency faults preserve the recorded economic values.
    pair = copy.deepcopy([facts['fills'][0], next(fill for fill in facts['fills']
        if fill['parent_id'] != facts['fills'][0]['parent_id'])])
    pair[0].update(account='a:b', parent_id='c')
    pair[1].update(account='a', parent_id='b:c')
    assert _cost_statistics._estimate(pair, CUTOFF, 'USDC')['parents'] == 2
    foreign = copy.deepcopy(facts)
    for fill in foreign['fills']:
        fill['fee_currency'] = 'foreign-unit-metadata-fault'
    foreign_report = CostCalibration.from_history(_write_fault(tmp_path / 'fault-fee-unit.json', foreign), cutoff_ns=CUTOFF).report()
    for side in ('buy', 'sell'):
        assert foreign_report['estimates'][side]['fee_bps'] is None
        assert foreign_report['estimates'][side]['known_builder_fee'] == '0'
        assert foreign_report['estimates'][side]['builder_fee_present_legs'] == 0
    for receipt in ('received_at_ns', 'reference_received_at_ns', 'fee_received_at_ns'):
        fault = copy.deepcopy(facts)
        fault['fills'][0][receipt] = 0
        with pytest.raises(ValueError, match='cannot precede'):
            CostCalibration.from_history(_write_fault(tmp_path / f'fault-early-{receipt}.json', fault), cutoff_ns=0)
    for label in ('benchmark_only', 'one_wrong_kind'):
        fault = copy.deepcopy(facts)
        fill_sha = fault['fills'][0]['source_asset_sha256']
        for asset in fault['source']['assets']:
            if label == 'benchmark_only' or asset['sha256'] == fill_sha:
                asset['kind'] = 'benchmark'
        with pytest.raises(ValueError, match='fills-kind source asset'):
            CostCalibration.from_history(_write_fault(tmp_path / f'fault-{label}.json', fault), cutoff_ns=CUTOFF)
    late = CostCalibration.from_history(path, cutoff_ns=CUTOFF - 1).report()
    assert late['support']['disposition'] == 'unavailable'
    assert late['estimates']['slip_bps'] is None
    assert 'required_evidence_after_cutoff' in late['support']['reasons']
    assert late['support']['admitted_legs'] < 363
    for label in ('duplicate', 'deleted'):
        fault = copy.deepcopy(facts)
        if label == 'duplicate':
            fault['fills'].append(copy.deepcopy(fault['fills'][0]))
        else:
            fault['fills'].pop()
        with pytest.raises(ValueError, match=r'Duplicate|Selected fill count'):
            CostCalibration.from_history(_write_fault(tmp_path / f'fault-{label}.json', fault), cutoff_ns=CUTOFF)
    fault = copy.deepcopy(facts)
    fault['source']['reference_max_age_ns'] = 60_000_000_000
    with pytest.raises(ValueError, match='record keys'):
        CostCalibration.from_history(_write_fault(tmp_path / 'fault-method-age.json', fault), cutoff_ns=CUTOFF)

    maker_path, maker_facts = _history(tmp_path, 'maker-rebates.json')
    assert all(Decimal(fill['fee']) < 0 for fill in maker_facts['fills'])
    maker = CostCalibration.from_history(maker_path, cutoff_ns=CUTOFF).report()
    assert Decimal(maker['estimates']['provided_maker_sample']['fee_bps']) < 0
    assert maker['support']['disposition'] == 'unavailable'
    diagnostic_path, diagnostic = _history(tmp_path, 'sensitivity-one-minute.json')
    buy = [fill for fill in diagnostic['fills'] if fill['side'] == 'buy']
    numerator = sum(Decimal(fill['quote_notional']) - Decimal(fill['quantity']) * Decimal(fill['reference_price']) for fill in buy)
    denominator = sum(Decimal(fill['quantity']) * Decimal(fill['reference_price']) for fill in buy)
    assert 10_000 * numerator / denominator == Decimal('-6.512474561429670499130878807')
    diagnostic_report = CostCalibration.from_history(diagnostic_path, cutoff_ns=CUTOFF).report()
    assert diagnostic_report['method'] == 'minute_midpoint_1s_v1'
    assert diagnostic_report['support']['selected_legs'] < len(diagnostic['fills'])
    assert diagnostic_report['estimates']['slip_bps'] is None

    real_single_day = copy.deepcopy(facts)
    start, end = real_single_day['source']['windows'][0]
    real_single_day['source']['windows'] = [[start, end]]
    real_single_day['source']['identity'] += ':report-only-june1'
    real_single_day['fills'] = [fill for fill in real_single_day['fills'] if start <= fill['time_ns'] < end]
    real_single_day['source_window_summary']['selected_legs'] = len(real_single_day['fills'])
    thin = CostCalibration.from_history(_write_fault(tmp_path / 'real-single-day.json', real_single_day), cutoff_ns=CUTOFF).report()
    assert 'thin_observed_support:sell' in thin['support']['reasons']
    assert 'negative_price_constant:sell' in thin['support']['reasons']
    assert thin['estimates']['slip_bps'] is None


def _assert_canonical_identity(recorded):
    identity = _cost_identity.digest
    assert identity(recorded) != identity(recorded.select(recorded.columns[::-1]))
    assert identity(recorded) != identity(recorded.rename({'side': 'declared_side'}))
    # These are encoding controls, not additional market observations.
    pairs = pl.concat([recorded, recorded.with_columns(pl.lit(None, dtype=pl.String).alias('side'))])
    assert identity(pairs) != identity(pairs.reverse())
    primitive = [pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
    assert len({identity(recorded.select(pl.col('is_taker').cast(dtype))) for dtype in primitive}) == len(primitive)
    clock = recorded.select(pl.col('time_ns').cast(pl.Datetime('ns')).alias('clock'))
    types = [pl.Date, pl.Time, *(pl.Datetime(unit, zone) for unit in ('ns', 'us', 'ms') for zone in (None, 'UTC')),
             *(pl.Duration(unit) for unit in ('ns', 'us', 'ms'))]
    frames = [clock.select(pl.col('clock').cast(dtype)) if dtype != pl.Time
              else clock.select(pl.col('clock').dt.time()) for dtype in types if not isinstance(dtype, pl.Duration)]
    elapsed = recorded.select((pl.col('time_ns') - pl.col('time_ns')).alias('clock'))
    frames.extend(elapsed.select(pl.col('clock').cast(dtype)) for dtype in types if isinstance(dtype, pl.Duration))
    assert len({identity(frame) for frame in frames}) == len(types)
    assert identity({'a': recorded, 'b': (True,)}) == identity({'b': (True,), 'a': recorded.clone()})
    assert identity(Decimal('1.0')) != identity(1.0) != identity(1) != identity(True)
    for dtype in (pl.Categorical, pl.List(pl.String), pl.Struct({'side': pl.String})):
        column = pl.col('side').cast(dtype) if dtype == pl.Categorical else (pl.concat_list('side') if isinstance(dtype, pl.List) else pl.struct('side'))
        with pytest.raises(ValueError, match='Unsupported canonical frame dtype'):
            identity(recorded.select(column))
    for value in (float('nan'), float('inf'), Decimal('NaN'), recorded.select(pl.lit(float('nan')).alias('price'))):
        with pytest.raises(ValueError, match='Nonfinite'):
            identity(value)
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match='Cyclic'):
        identity(cyclic)
    for value in ({1: 'non-string key'}, object()):
        with pytest.raises(TypeError):
            identity(value)
