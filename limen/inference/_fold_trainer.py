from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import cast

import polars as pl

from limen.backtest.trade_contract import JsonValue
import limen.experiment._fold_evidence as evidence
import limen.experiment._holdout_folds as protection
from limen.experiment._walk_forward_run import WalkForwardRun
from limen.experiment.manifest_core import Manifest
import limen.inference._fold_metrics as _metrics
import limen.inference._fold_sensor as _guard
from limen.inference.errors import ReconstructionError
from limen.inference.sensor import Sensor
from limen.sfd.reference_architecture.base import ReferenceModel

__all__: list[str] = []
_VALIDATION_RATIO_FIELDS = 2


@dataclass
class Source:
    metadata: dict[str, object]
    round_data: dict[str, dict[str, object]]
    manifest: Manifest
    data: pl.DataFrame
    experiment_dir: Path
    yaml_reference: dict[str, object]
    manifest_id: str | None


def restore_ratio(manifest: Manifest, metadata: Mapping[str, object]) -> None:
    ratios = cast(list[int], metadata['walk_forward_validation_ratio'])
    if len(ratios) != _VALIDATION_RATIO_FIELDS or any(type(value) is not int for value in ratios):
        raise ValueError('invalid saved validation ratio')
    _ = manifest.set_split_config(ratios[0], ratios[1], manifest.split_config[2])


def train(owner: object, ids: list[str], fold: int | None) -> list[Sensor] | None:
    attrs = vars(owner)
    host = Source(cast(dict[str, object], attrs['_metadata']), cast(dict[str, dict[str, object]], attrs['_round_data']),
                  cast(Manifest, attrs['_manifest']), cast(pl.DataFrame, attrs['_data']), cast(Path, attrs['_experiment_dir']),
                  cast(dict[str, object], attrs['_yaml_reference']), cast(str | None, attrs['_manifest_id']))
    if fold is not None and type(fold) is not int:
        raise ValueError('fold must be a literal integer')
    config = host.manifest.split_walk_forward
    saved = host.metadata.get('split_walk_forward')
    if config is None and saved is None:
        if fold is not None:
            raise ValueError('fold is only valid for walk-forward')
        return None
    if fold is None:
        raise ValueError('walk-forward Trainer requires train(..., fold=k)')
    n_folds = cast(Mapping[str, object], saved).get('n_folds') if isinstance(saved, Mapping) else None
    if type(n_folds) is int and not 0 <= fold < n_folds:
        raise ValueError('fold out of range')
    if config is None or saved != config.as_dict():
        raise ReconstructionError('walk-forward metadata declaration changed; rerun with a complete fold witness')
    missing = [pid for pid in ids if pid not in host.round_data]
    if missing:
        raise ValueError(f'Permutation IDs not found in round_data: {missing}')
    return [_one(host, pid, fold) for pid in ids]


def _one(host: Source, pid: str, fold: int) -> Sensor:
    try:
        expected = host.round_data[pid]
        context = evidence.read_fold_context(host.metadata, expected, fold)
        campaign = cast(Mapping[str, object], context['campaign'])
        original = cast(Mapping[str, object], campaign['source'])
        trial = cast(Mapping[str, object], context['trial'])
        witness = cast(Mapping[str, object], context['fold_witness'])
        for entry in host.round_data.values():
            other = evidence.checked(entry.get('fold_promotion'))
            if other.get('campaign_source_digest') != original['digest']:
                raise ValueError('competing trial campaign source identity changed')
        manifest = copy.deepcopy(host.manifest)
        restore_ratio(manifest, host.metadata)
        if list(manifest.split_config[:2]) != trial['validation_ratio']:
            raise ValueError('validation ratio changed')
        params = dict(cast(Mapping[str, object], expected['round_params']))
        if evidence.preparation(manifest, params) != trial['preparation']:
            raise ValueError('resolved preparation/architecture/source declaration changed')
        if getattr(manifest, 'decoder_lookback', 1) != 1:
            raise ValueError('native inference requires decoder_lookback=1')
        runner = WalkForwardRun(manifest, host.data, host.experiment_dir, promotion=True)
        actual_campaign = evidence.checked(runner.promotion)
        if actual_campaign != {key: value for key, value in campaign.items() if key != 'origin'}:
            raise ValueError('original source identity/extent changed; pin end_date_limit or supply the recorded data')
        prepared_trial = runner.prepare(host.data, params)
        if prepared_trial['_fold_trial'] != trial:
            raise ValueError('selected source values/validation/preparation changed before preprocessing fit')
        vars(manifest)['_walk_forward_fold'] = fold
        manifest.pre_split_data_selector = None
        evidence.configure(manifest, prepared_trial, fold)
        capture = cast(dict[str, object], vars(manifest)['_fold_capture'])
        capture['expected'] = witness
        raw = cast(pl.DataFrame, prepared_trial['_walk_forward_raw'])
        with protection.trainer(manifest, expected, fold):
            data = manifest.prepare_data(raw, params)
            partitions = evidence.checked(data.get('_fold_partitions'))
            if any(partitions[key] != witness[key] for key in ('raw', 'retained')):
                raise ValueError('fold partitions changed before model fit')
            raw_parts = cast(Mapping[str, Mapping[str, object]], partitions['raw'])
            support = [end - 1 for name in ('fit', 'validation') for _, end in cast(list[list[int]], raw_parts[name]['segments'])]
            times = cast(list[int], data['_fold_selected_times'])
            cutoff = max(times[index] for index in support) + cast(int, original['interval_ns'])
            if cutoff != witness['fitting_cutoff_ns']:
                raise ValueError('fitting cutoff disagrees with raw fit/validation support')
            data['_record_execution'] = True
            data['_record_model_outputs'] = bool(host.metadata.get('record_model_outputs'))
            result = manifest.run_model(data, params)
            model = result.get('_model')
            if model is None:
                raise ValueError('selected fold did not return a fitted native inference model')
            model = cast(ReferenceModel, model)
            selected = cast(Mapping[str, object], cast(list[object], expected['folds'])[fold])
            policy = cast(Mapping[str, bool], host.metadata['recording'])
            _metrics.bindings(selected, data, witness, model, policy)
            _metrics.validate(cast(Mapping[str, object], selected['results']), result, model.deterministic)
            _metrics.stable_model(model, data)
            context['results'] = copy.deepcopy(selected['results'])
            return _FoldSensor(context, yaml_reference=host.yaml_reference, model=model,
                                        fitted_params=cast(dict[str, object], data.get('_fitted_params', {})),
                                        round_params=params, permutation_id=pid, manifest_id=host.manifest_id,
                                        trade_contract=cast(Mapping[str, JsonValue] | None, data.get('trade_contract')))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ReconstructionError(f'Permutation {pid}, fold {fold}: {exc}') from exc


class _FoldSensor(_guard.FoldView, Sensor):
    def __init__(self, context: dict[str, object], *, yaml_reference: dict[str, object], model: ReferenceModel,
                 fitted_params: dict[str, object], round_params: dict[str, object], permutation_id: str,
                 manifest_id: str | None, trade_contract: Mapping[str, JsonValue] | None) -> None:
        _guard.FoldView.__init__(self, context)
        Sensor.__init__(self, yaml_reference, model, fitted_params, round_params, permutation_id, manifest_id, trade_contract=trade_contract)
