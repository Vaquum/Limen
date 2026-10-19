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
    artifact = tmp_path / 'calibration.json'
    expected_digest = calibration.digest  # Persist separately from the artifact in the caller specification.
    assert calibration.write(artifact) == calibration.write(artifact) == expected_digest
    with monkeypatch.context() as context:
        context.setattr(_cost_statistics, 'report', _no_estimation)
        restored = CostCalibration.read(artifact, expected_digest=expected_digest)
    assert restored.report() == calibration.report()
    report['estimates'].clear()
    assert calibration.report()['estimates']['buy']['parents'] == 142
    with pytest.raises(AttributeError, match='immutable'):
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
        with pytest.raises(ValueError, match='Duplicate|Selected fill count'):
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
