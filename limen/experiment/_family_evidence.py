import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl

import limen.experiment._round_storage as storage
from limen.backtest.trade_contract import JsonValue, contract_digest
from limen.yaml.config import round_trip_yaml
from limen.yaml.store import STORE_RELATIVE, canonical_manifest_id, is_full_manifest_id

__all__: list[str] = []


class EvidenceUnavailableError(ValueError):
    '''A required part of the recorded comparison evidence is absent.'''


class EvidenceIncompatibleError(ValueError):
    '''Present timestamp evidence cannot support a comparison.'''


@dataclass
class Trial:
    identity: dict[str, str]
    values: npt.NDArray[np.float64]
    candidate: str | None
    evaluation: str | None
    evidence: dict[str, object] | None
    report: dict[str, object]


def mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in cast(dict[object, object], value)):
        raise ValueError(f'{name} must be a mapping with string keys')
    return cast(dict[str, object], value)


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def chronology(metadata: Mapping[str, object]) -> datetime | None:
    value = metadata.get('created_at')
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('Recorded created_at must be an ISO timestamp')
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Recorded created_at must identify its timezone')
    return parsed.astimezone(timezone.utc)


def lineage(metadata: Mapping[str, object], project_root: Path) -> list[dict[str, object]]:
    current = metadata.get('manifest_id')
    reference = metadata.get('yaml_reference')
    if current is None:
        return []
    if not is_full_manifest_id(current):
        raise ValueError('Invalid recorded manifest_id')
    if reference is not None and canonical_manifest_id(mapping(reference, 'yaml_reference')) != current:
        raise ValueError('Recorded manifest_id differs from yaml_reference')
    chain: list[dict[str, object]] = []
    seen: set[str] = set()
    while current is not None:
        if not is_full_manifest_id(current) or current in seen:
            raise ValueError('Invalid or cyclic recorded manifest ancestry')
        seen.add(current)
        path = project_root / STORE_RELATIVE / (current.removeprefix('sha256:') + '.yaml')
        if not path.exists():
            chain.append({'manifest_id': current, 'available': False})
            break
        document = mapping(cast(object, round_trip_yaml().load(path.read_text())), str(path))
        recorded = mapping(document.get('lineage'), 'committed manifest lineage')
        if recorded.get('id') != current or canonical_manifest_id(document) != current:
            raise ValueError(f'Committed manifest identity mismatch: {path}')
        parent, committed = recorded.get('parent_id'), recorded.get('committed_at')
        if parent is not None and not isinstance(parent, str):
            raise ValueError('Recorded parent_id must be a manifest identity')
        if not isinstance(committed, str):
            raise ValueError('Recorded committed_at must be an ISO timestamp')
        chain.append({'manifest_id': current, 'parent_id': parent, 'committed_at': committed, 'available': True})
        current = parent
    return list(reversed(chain))


def _records(directory: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    path = storage.round_path(directory)
    if not path.exists():
        return records
    with storage.open_text(path) as lines:
        for line in lines:
            record = mapping(cast(object, json.loads(line)), 'recorded trial')
            trial = record.get('round_id')
            if not isinstance(trial, str) or trial in records:
                raise ValueError('Family report requires unique recorded round identities within each run')
            records[trial] = record
    return records


def _timestamps(value: object) -> list[str]:
    if value is None:
        raise EvidenceUnavailableError('Recorded ordered test timestamps are missing')
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in cast(list[object], value)):
        raise EvidenceIncompatibleError('Recorded test timestamps must be a nonempty ISO timestamp list')
    try:
        dates = [datetime.fromisoformat(item) for item in cast(list[str], value)]
    except ValueError as exc:
        raise EvidenceIncompatibleError('Recorded test timestamps must be ISO timestamps') from exc
    if len({item.tzinfo is None for item in dates}) != 1:
        raise EvidenceIncompatibleError('Recorded test timestamps mix zoned and unzoned observations')
    normalized = [item.astimezone(timezone.utc) if item.tzinfo is not None else item for item in dates]
    if normalized != sorted(set(normalized)):
        raise EvidenceIncompatibleError('Recorded test timestamps must be unique and increasing')
    return [item.isoformat() for item in normalized]


def _sample(fold: Mapping[str, object], count: int) -> dict[str, object]:
    alignment = fold.get('alignment')
    if alignment is None:
        raise EvidenceUnavailableError('Recorded fold alignment is missing')
    dates = _timestamps(mapping(alignment, 'fold alignment').get('test_datetimes'))
    if len(dates) != count:
        raise ValueError('Recorded test timestamps differ in length from fold returns')
    contract = fold.get('trade_contract')
    if contract is not None:
        declaration = mapping(contract, 'trade contract')
        if contract_digest(cast(dict[str, JsonValue], declaration)) != fold.get('trade_contract_digest'):
            raise ValueError('Recorded trade contract digest mismatch')
        policy = mapping(declaration.get('policy'), 'trade policy')
        product = mapping(policy.get('product'), 'trade product')
        sources = declaration.get('sources')
        if not isinstance(sources, list):
            raise EvidenceUnavailableError('Recorded execution sources are missing')
        execution = [mapping(item, 'trade source') for item in cast(list[object], sources)
                     if mapping(item, 'trade source').get('interpretation') == 'causal_recorded_prices']
        if not execution or any(not item.get('checksum') for item in execution):
            raise EvidenceUnavailableError('Recorded execution observation identity is missing')
        if any(declaration.get(key) is None for key in ('rule_version', 'partition', 'signal_rows')):
            raise EvidenceUnavailableError('Recorded execution rule, partition or signal availability is missing')
        signals = declaration['signal_rows']
        if not isinstance(signals, list) or len(cast(list[object], signals)) != count:
            raise ValueError('Recorded signal availability differs in length from fold returns')
        return {'dates': dates, 'semantics': 'signed_execution', 'rule_version': declaration.get('rule_version'),
                'product': {key: product[key] for key in ('kind', 'instrument', 'base_currency', 'quote_currency')},
                'rules': declaration.get('rules'), 'partition': declaration.get('partition'),
                'signal_rows': declaration.get('signal_rows'), 'sources': execution}
    market = fold.get('market')
    if market is None:
        raise EvidenceUnavailableError('Recorded snapshot market observations are missing')
    returns = mapping(market, 'snapshot market').get('ret')
    if not isinstance(returns, list) or len(cast(list[object], returns)) != count:
        raise ValueError('Recorded snapshot market observations differ in length from fold returns')
    for value in cast(list[object], returns):
        if value is not None and (type(value) not in (int, float) or not np.isfinite(cast(float, value))):
            raise ValueError('Recorded market returns must be finite numbers or recorded nulls')
    if all(value is None for value in cast(list[object], returns)):
        raise EvidenceUnavailableError('Recorded snapshot market observations contain no observed returns')
    return {'dates': dates, 'semantics': 'snapshot', 'market_returns': returns}


def _evidence(metadata: Mapping[str, object], record: Mapping[str, object], values: npt.NDArray[np.float64]) -> tuple[str, str, dict[str, object]]:
    reference = metadata.get('yaml_reference')
    if reference is None:
        raise EvidenceUnavailableError('Recorded manifest configuration is missing')
    sfd = mapping(mapping(reference, 'yaml_reference').get('sfd'), 'recorded SFD')
    manifest = mapping(sfd.get('manifest'), 'recorded manifest')
    params = record.get('round_params')
    if params is None:
        raise EvidenceUnavailableError('Recorded round parameters are missing')
    params = mapping(params, 'round parameters')
    source = manifest.get('data_source')
    if source is not None:
        source = mapping(source, 'data source')
        source_params = source.get('params')
        source = {**source, 'params': mapping({} if source_params is None else source_params, 'data-source parameters')}
    folds = record.get('folds')
    if folds is None:
        raise EvidenceUnavailableError('Recorded fold evidence is missing')
    if not isinstance(folds, list) or not folds:
        raise ValueError('Recorded folds must be a nonempty list')
    decoded_folds: list[tuple[dict[str, object], int]] = []
    recorded: list[float] = []
    for value in cast(list[object], folds):
        fold = mapping(value, 'recorded fold')
        net = fold.get('net_returns')
        if net is None:
            raise EvidenceUnavailableError('Recorded fold net returns are missing')
        if not isinstance(net, list) or not net or any(type(item) not in (int, float) for item in cast(list[object], net)):
            raise ValueError('Recorded fold net returns must be a nonempty numeric list')
        recorded.extend(cast(list[float], net))
        decoded_folds.append((fold, len(cast(list[object], net))))
    if not np.array_equal(np.asarray(recorded, dtype=np.float64), values):
        raise ValueError('Recorded fold returns differ from trial_returns.parquet')
    samples = [_sample(fold, count) for fold, count in decoded_folds]
    dates = [stamp for sample in samples for stamp in cast(list[str], sample['dates'])]
    parsed = [datetime.fromisoformat(stamp) for stamp in dates]
    if len({stamp.tzinfo is None for stamp in parsed}) != 1 or parsed != sorted(set(parsed)):
        raise EvidenceIncompatibleError('Recorded test folds overlap or differ in timestamp convention')
    candidate = {key: value for key, value in sfd.items() if key != 'params'}
    candidate['manifest'] = {key: value for key, value in manifest.items() if key != 'acceptance'}
    summary: dict[str, object] = {'source': source, 'source_verified': False,
        'declared_manifest': manifest, 'round_params': params,
        'split_walk_forward': metadata.get('split_walk_forward'),
        'walk_forward_validation_ratio': metadata.get('walk_forward_validation_ratio'),
        'recording': metadata.get('recording'), 'record_execution': metadata.get('record_execution'),
        'pbo_unavailable': 'Snapshot artifacts do not retain complete open/high/low execution identity' if any(sample['semantics'] == 'snapshot' for sample in samples) else None,
        'folds': [
        {'n_bars': len(cast(list[str], sample['dates'])),
         'first_test_datetime': cast(list[str], sample['dates'])[0], 'last_test_datetime': cast(list[str], sample['dates'])[-1],
         'missing_datetimes': mapping(fold.get('alignment'), 'fold alignment').get('missing_datetimes'),
         'timestamps_digest': digest(sample['dates']), 'return_semantics': sample['semantics'],
         'market_evidence_digest': digest({key: value for key, value in sample.items() if key != 'dates'}),
         **{key: sample[key] for key in ('product', 'rule_version', 'partition') if key in sample}}
        for sample, (fold, _) in zip(samples, decoded_folds, strict=True)]}
    return digest({'sfd': candidate, 'params': params, 'limen_version': metadata.get('limen_version'),
                   'geometry': metadata.get('split_walk_forward'), 'validation': metadata.get('walk_forward_validation_ratio')}), digest(samples), summary


def read_run(directory: Path, metadata: dict[str, object], project_root: Path) -> tuple[dict[str, object], list[Trial]]:
    created = chronology(metadata)
    ancestry = lineage(metadata, project_root)
    errors: list[str] = []
    report: dict[str, object] = {'directory': str(directory), 'created_at': None if created is None else created.isoformat(),
                                 'manifest_id': metadata.get('manifest_id'), 'lineage': ancestry, 'errors': errors,
                                 **{key: metadata.get(key) for key in ('split_walk_forward', 'walk_forward_validation_ratio', 'recording', 'record_execution', 'limen_version')},
                                 'track_coverage_known': False, 'terminal_trials': None, 'n_failed_trials': None}
    records = _records(directory)
    terminal: list[dict[str, object]] | None = None
    if (csv_path := directory / 'results.csv').exists():
        with csv_path.open(newline='') as stream:
            terminal = [{'round_index': index, 'trial': row['id'], 'error': row.get('strict_mode_error') or None}
                        for index, row in enumerate(csv.DictReader(stream))]
        report.update(terminal_trials=terminal, n_failed_trials=sum(row['error'] is not None for row in terminal))
        if records and [(row['round_index'], row['trial']) for row in terminal if row['error'] is None] != [(record.get('_round_index'), trial) for trial, record in records.items()]:
            raise storage.RoundStorageError('Recorded rounds differ from successful terminal CSV occurrences')
    if created is None:
        errors.append('Recorded run chronology is unavailable')
    if not ancestry or not all(item['available'] for item in ancestry):
        errors.append('Recorded manifest ancestry is incomplete')
    path = directory / 'trial_returns.parquet'
    if not path.exists():
        errors.append('trial_returns.parquet is missing; recorded trial coverage is unknown')
        return report, []
    frame = pl.read_parquet(path)
    if frame.schema != {'trial': pl.String, 'bar': pl.Int64, 'net_return': pl.Float64}:
        raise ValueError('Family report requires the recorded trial_returns.parquet schema')
    identifiers = cast(list[object], frame['trial'].unique(maintain_order=True).to_list())
    if not all(isinstance(trial, str) for trial in identifiers):
        raise ValueError('Recorded trial identities must be strings')
    identities = cast(list[str], identifiers)
    identity_set = set(identities)
    if terminal is not None and {row['trial'] for row in terminal if row['error'] is None} != identity_set:
        raise storage.RoundStorageError('Recorded return tracks differ from successful terminal CSV trials')
    report['track_coverage_known'] = True
    if records.keys() - identity_set:
        raise ValueError('Recorded trials are absent from trial_returns.parquet')
    ordered = [*records, *(trial for trial in identities if trial not in records)]
    trials: list[Trial] = []
    for trial in ordered:
        rows = frame.filter(pl.col('trial') == trial).sort('bar')
        if rows['bar'].to_list() != list(range(rows.height)):
            raise ValueError('Family report requires unique contiguous per-trial bar identities')
        values = np.asarray(rows['net_return'].to_numpy(), dtype=np.float64)
        if not values.size or not np.isfinite(values).all():
            raise ValueError('Family report requires nonempty finite recorded return tracks')
        identity = {'directory': str(directory), 'trial': trial}
        entry: dict[str, object] = {**identity, 'candidate': None, 'comparison': None, 'repeat_of': None,
                                   'evidence_match_of': None, 'repeat_identity': 'unproven', 'unavailable': None,
                                   'round_index': records.get(trial, {}).get('_round_index'), 'disposition': 'included', 'evidence': None}
        candidate = evaluation = None
        summary: dict[str, object] | None = None
        try:
            if trial not in records:
                raise EvidenceUnavailableError('Recorded trial/fold evidence is missing')
            candidate, evaluation, summary = _evidence(metadata, records[trial], values)
            entry['candidate'] = candidate
            entry['evidence'] = summary
        except (EvidenceUnavailableError, EvidenceIncompatibleError) as exc:
            entry['unavailable'] = str(exc)
            entry['disposition'] = 'missing_evidence' if isinstance(exc, EvidenceUnavailableError) else 'incompatible_evidence'
            entry['recording'] = {'recording': metadata.get('recording'), 'record_execution': metadata.get('record_execution')}
        trials.append(Trial(identity, values, candidate, evaluation, summary, entry))
    if not trials:
        errors.append('No successful recorded trial tracks are present')
    return report, trials
