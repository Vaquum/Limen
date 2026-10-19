from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from hashlib import sha256
import json
import math
from typing import TYPE_CHECKING, Protocol, cast

import polars as pl

from limen.experiment._fold_fingerprint import FoldEvidenceError, source, validate_schema
from limen.experiment._fold_partitions import validate_partition
from limen.experiment._holdout_research import prepare_walk_forward
from limen.experiment._resolve_trade_policy import resolve_json

if TYPE_CHECKING:
    from limen.experiment.manifest_core import Manifest

__all__: list[str] = []
_OUTPUT_FIELDS = 2
_DIGEST_LENGTH = 64


class Host(Protocol):
    manifest: Manifest
    promotion: dict[str, object] | None


class Record(Protocol):
    def __call__(self, fold: int, data: Mapping[str, object], result: Mapping[str, object], *, record_execution: bool, record_outputs: bool) -> dict[str, object]: ...


def qualified(value: object) -> str:
    return f"{getattr(value, '__module__', None)}.{getattr(value, '__qualname__', None)}"


def preparation(manifest: Manifest, params: Mapping[str, object]) -> dict[str, object]:
    data_source = manifest.data_source_config
    formation = manifest.bar_formation
    return {'architecture': qualified(manifest.architecture_function),
            'data_source': None if data_source is None else {'method': qualified(data_source.method), 'params': resolve_json(data_source.params, params)},
            'bar_type': params.get('bar_type', 'base'),
            'formation': None if formation is None else {'function': qualified(formation[0]), 'params': resolve_json(formation[1], params)},
            'decoder_lookback': getattr(manifest, 'decoder_lookback', 1)}


def start(manifest: Manifest, raw: pl.DataFrame, enabled: bool | dict[str, object]) -> dict[str, object] | None:
    if not enabled:
        return None
    if isinstance(enabled, dict):
        return copy.deepcopy(enabled)
    try:
        config = manifest.data_source_config
        seconds = None if config is None else config.params.get('kline_size', config.params.get('klines_size'))
        witness = source(raw, seconds)
        return {'version': 1, 'source': witness, 'selection_cutoff_ns': cast(int, witness['last_ns']) + cast(int, witness['interval_ns'])}
    except FoldEvidenceError as exc:
        return exc.record()


def prepare(host: Host, raw: pl.DataFrame, params: Mapping[str, object] | None,
            resolve_params: Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]],
            requires_validation: Callable[..., bool]) -> dict[str, object]:
    data = prepare_walk_forward(host, raw, params, resolve_params, requires_validation)
    if host.promotion is not None:
        try:
            campaign = checked(host.promotion)
            original = cast(dict[str, object], campaign['source'])
            selected = original if host.manifest.pre_split_data_selector is None else source(cast(pl.DataFrame, data['_walk_forward_raw']), cast(int, original['interval_ns']) / 1_000_000_000)
            data['_fold_trial'] = {'version': 1, 'campaign_source_digest': original['digest'], 'selected_source': selected,
                                   'validation_ratio': list(host.manifest.split_config[:2]), 'preparation': preparation(host.manifest, params or {})}
        except FoldEvidenceError as exc:
            data['_fold_trial'] = exc.record()
    return data


def configure(manifest: Manifest, data: Mapping[str, object], fold: int) -> None:
    if '_fold_trial' in data:
        trial = cast(Mapping[str, object], data['_fold_trial'])
        prep = cast(Mapping[str, object], trial.get('preparation', {}))
        vars(manifest)['_fold_capture'] = {'fold': fold, 'trial': trial, 'formed': prep.get('bar_type', 'base') != 'base' and prep.get('formation') is not None}


def model_witness(model: object) -> dict[str, object]:
    if model is None or not callable(getattr(model, 'predict', None)):
        raise FoldEvidenceError('inference_model', 'The architecture did not return a fitted native prediction model; rerun with a reconstructable inference contract')
    supplied: object = getattr(type(model), 'inference_output', None)
    if supplied is not None and not isinstance(supplied, tuple):
        raise FoldEvidenceError('inference_model', 'inference_output must be an immutable tuple')
    declaration = cast(tuple[object, ...] | None, supplied)
    output: dict[str, object] | None
    if declaration is not None and (len(declaration) != _OUTPUT_FIELDS or declaration[0] not in ('binary', 'continuous', 'target_exposure') or type(declaration[1]) is not bool):
        raise FoldEvidenceError('inference_model', 'inference_output must be an immutable (kind, bool_probabilities) class declaration')
    output = None if declaration is None else {'kind': declaration[0], 'probabilities': declaration[1]}
    mode = getattr(model, 'prediction_mode', None)
    if mode not in ('binary', 'target_exposure') or type(getattr(model, 'deterministic', None)) is not bool:
        raise FoldEvidenceError('inference_model', 'Fitted model lacks native prediction_mode/deterministic declarations')
    return {'class': qualified(type(model)), 'prediction_mode': mode, 'output_contract': output}


def record(make_record: Record, fold: int, data: Mapping[str, object], result: Mapping[str, object], *, record_execution: bool, record_outputs: bool) -> dict[str, object]:
    entry = make_record(fold, data, result, record_execution=record_execution, record_outputs=record_outputs)
    if '_fold_trial' not in data:
        return entry
    try:
        trial = checked(data['_fold_trial'])
        parts = checked(data.get('_fold_partitions'))
        raw = cast(Mapping[str, Mapping[str, object]], parts['raw'])
        positions = [end - 1 for name in ('fit', 'validation') for _, end in cast(list[list[int]], raw[name]['segments'])]
        selected_times = cast(list[int], data['_fold_selected_times'])
        selected = cast(Mapping[str, object], trial['selected_source'])
        entry['fold_promotion'] = {'version': 1, 'fold': fold, 'n_folds': data['_fold_n_folds'], **parts,
                                   'fitting_cutoff_ns': max(selected_times[index] for index in positions) + cast(int, selected['interval_ns']),
                                   'fitted_threshold': getattr(result.get('_model'), '_calibration_threshold', None),
                                   'inference_model': model_witness(result.get('_model'))}
    except (FoldEvidenceError, IndexError) as exc:
        error = exc if isinstance(exc, FoldEvidenceError) else FoldEvidenceError('row_membership', f'Fold support escaped its selected source: {exc}')
        entry['fold_promotion'] = error.record()
    return entry


def add_metadata(metadata: dict[str, object], host: object, workers: int) -> None:
    promotion = getattr(host, 'promotion', None)
    if promotion is not None:
        metadata['fold_promotion'] = {**cast(Mapping[str, object], promotion), 'origin': 'parallel' if workers > 1 else 'sequential'} if 'unavailable_reason' not in promotion else promotion


def _versioned(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FoldEvidenceError('row_membership', 'Missing fold witness; rerun with version 1 evidence')
    record = cast(dict[str, object], value)
    if type(record.get('version')) is not int or record.get('version') != 1:
        raise FoldEvidenceError('row_membership', 'Unsupported fold witness; rerun with version 1 evidence')
    if 'unavailable_reason' in record:
        reason = record['unavailable_reason']
        if set(record) != {'version', 'unavailable_reason'} or not isinstance(reason, dict):
            raise ValueError('Invalid unavailable fold witness; rerun')
        fields = cast(dict[str, object], reason)
        if set(fields) != {'code', 'detail'} or fields['code'] not in ('clock', 'dtype', 'row_membership', 'inference_model') or not isinstance(fields['detail'], str) or not fields['detail']:
            raise ValueError('Invalid unavailable fold witness reason; rerun')
    return record


def checked(value: object) -> dict[str, object]:
    record = _versioned(value)
    if 'unavailable_reason' in record:
        reason = cast(Mapping[str, object], record['unavailable_reason'])
        raise FoldEvidenceError(str(reason['code']), str(reason['detail']))
    return record


def _source(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError('fold witness source must be an object; rerun')
    item = cast(dict[str, object], value)
    if set(item) != {'rows', 'schema', 'digest', 'first_ns', 'last_ns', 'interval_ns'} or any(type(item[key]) is not int for key in ('rows', 'first_ns', 'last_ns', 'interval_ns')):
        raise ValueError('fold witness source fields are incomplete; rerun')
    digest = item['digest']
    if not isinstance(digest, str) or len(digest) != _DIGEST_LENGTH or any(char not in '0123456789abcdef' for char in digest) or cast(int, item['rows']) <= 0 or cast(int, item['interval_ns']) <= 0 or cast(int, item['first_ns']) > cast(int, item['last_ns']):
        raise ValueError('fold witness source identity is invalid; rerun')
    validate_schema(item['schema'])
    return item


def read_fold_context(metadata: Mapping[str, object], round_entry: Mapping[str, object], fold: int) -> dict[str, object]:
    recording = metadata.get('recording')
    if not isinstance(recording, Mapping) or cast(Mapping[str, object], recording).get('fold_details') is not True:
        raise ValueError('Fold promotion requires uel.recording.fold_details: true; rerun')
    policy = cast(Mapping[str, object], recording)
    campaign, trial = checked(metadata.get('fold_promotion')), checked(round_entry.get('fold_promotion'))
    if set(campaign) != {'version', 'origin', 'source', 'selection_cutoff_ns'} or set(trial) != {'version', 'campaign_source_digest', 'selected_source', 'validation_ratio', 'preparation'}:
        raise ValueError('fold witness campaign/trial fields are incomplete; rerun')
    original, selected = _source(campaign.get('source')), _source(trial.get('selected_source'))
    config = cast(Mapping[str, object], metadata['split_walk_forward'])
    n_folds = config['n_folds']
    folds = round_entry.get('folds')
    if type(n_folds) is not int or not isinstance(folds, list) or len(cast(list[object], folds)) != n_folds:
        raise ValueError('fold witness count disagrees with split_walk_forward; rerun')
    if campaign.get('origin') not in ('sequential', 'parallel') or campaign.get('selection_cutoff_ns') != cast(int, original['last_ns']) + cast(int, original['interval_ns']) or trial.get('campaign_source_digest') != original['digest'] or selected['interval_ns'] != original['interval_ns']:
        raise ValueError('fold witness campaign/source/selection cutoff mismatch; rerun')
    if trial.get('validation_ratio') != metadata.get('walk_forward_validation_ratio') or not isinstance(trial.get('preparation'), dict):
        raise ValueError('fold witness validation ratio/preparation mismatch; rerun')
    entries = cast(list[Mapping[str, object]], folds)
    for index, entry in enumerate(entries):
        witness = _versioned(entry.get('fold_promotion'))
        if type(entry.get('fold')) is not int or entry.get('fold') != index:
            raise ValueError('fold witness identity/count mismatch; rerun')
        if 'unavailable_reason' not in witness and (type(witness.get('fold')) is not int or type(witness.get('n_folds')) is not int or witness.get('fold') != index or witness.get('n_folds') != n_folds):
            raise ValueError('fold witness identity/count mismatch; rerun')
    entry = entries[fold]
    witness = checked(entry['fold_promotion'])
    if set(witness) != {'version', 'fold', 'n_folds', 'raw', 'retained', 'fitting_cutoff_ns', 'fitted_threshold', 'inference_model'}:
        raise ValueError('fold witness fitted fields are incomplete; rerun')
    threshold = witness['fitted_threshold']
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
        raise ValueError('fold witness fitted threshold must be finite or absent')
    if type(witness.get('fitting_cutoff_ns')) is not int or cast(int, witness['fitting_cutoff_ns']) > cast(int, campaign['selection_cutoff_ns']) or not isinstance(witness.get('inference_model'), dict):
        raise ValueError('fold witness fitted inference context is incomplete; rerun')
    for key in ('raw', 'retained'):
        groups = witness.get(key)
        if not isinstance(groups, dict) or set(cast(dict[str, object], groups)) != {'fit', 'validation', 'test'}:
            raise ValueError('fold witness partition context is incomplete; rerun')
        for partition in cast(dict[str, object], groups).values():
            validate_partition(partition, cast(int, selected['rows']), raw_rows=key == 'raw')
    payload = {'version': 1, 'manifest_id': metadata['manifest_id'], 'permutation_id': round_entry['round_id'],
               'fold': fold, 'campaign': campaign, 'trial': trial, 'fold_witness': witness,
               'trade_contract_digest': entry.get('trade_contract_digest'), 'learning_binding': entry.get('learning_binding'),
               'recording': {key: bool(policy[key]) for key in ('predictions', 'alignment', 'trade_ledger', 'fold_details')},
               'record_model_outputs': bool(metadata.get('record_model_outputs'))}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return copy.deepcopy({**payload, 'provenance_digest': digest})
