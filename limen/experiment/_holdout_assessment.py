from __future__ import annotations

import copy
from collections.abc import Callable
import json
import logging
import math
from pathlib import Path
from threading import get_ident
import time
from typing import cast
from typing_extensions import override

import polars as pl

from limen.backtest._snapshot_execution import snapshot_with_execution
from limen.cohort import Cohort
from limen.experiment import _holdout_attempts as attempts
from limen.experiment import _holdout_history as history
from limen.experiment import _holdout_report as reports
from limen.experiment._holdout_freeze import FrozenHoldoutSelection, digest, members, owned_copy, provenance, state_digest, unchanged, frozen_parts
from limen.experiment._holdout_prediction import readonly_prediction
from limen.experiment._holdout_source import SourceEvidence, load_admitted_source, source_spec
from limen.experiment.manifest_core import MLManifest
from limen.inference.sensor import BarPrediction, Sensor

_aggregate = cast(Callable[[Cohort, list[BarPrediction]], BarPrediction], vars(Cohort)['_aggregate_bar_predictions'])

class _Failures(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.member_id: str | None = None
        self.bar_ns: int | None = None
        self.message: str | None = None
        self.thread = get_ident()

    @override
    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread:
            self.message = record.getMessage()

    def schema(self, message: str) -> None:
        self.message = message


def _metadata(selection: Sensor | Cohort, root: Path, declaration: dict[str, object], spec: dict[str, object]) -> list[dict[str, object]]:
    proofs: list[dict[str, object]] = []
    sources: list[object] = []
    binding = history.ProjectBinding(root, cast(str, declaration['project_id']), cast(int, declaration['revision']), None, cast(int, declaration['interval_seconds']))
    for member, declared in zip(members(selection), cast(list[dict[str, object]], spec['members']), strict=True):
        proof = provenance(member)
        if proof['project_id'] != binding.project_id or proof['round_id'] != member.permutation_id or history.utc_day(cast(str, declared['test_end'])) > cast(int, declaration['start_ns']):
            raise ValueError('Holdout selection lineage/project/test_end disagrees with final window')
        trainer = history.research_record(binding, cast(str, proof['research_record_id']))
        experiment = history.research_record(binding, cast(str, proof['experiment_record_id']))
        if trainer['state'] != 'completed' or trainer['role'] != 'trainer' or experiment['state'] != 'completed' or experiment['role'] != 'experiment' or trainer['round_id'] != proof['round_id'] or experiment['round_id'] != proof['round_id'] or json.loads(cast(str, trainer['provenance_json'])) != proof:
            raise ValueError('Holdout requires matching completed Trainer and source experiment records')
        parent = cast(dict[str, object], json.loads(cast(str, experiment['provenance_json'])))
        for retained in (proof, parent):
            admitted = cast(dict[str, object], cast(dict[str, object], retained['source'])['admitted'])
            last_close = admitted['last_close_ns']
            if type(last_close) is not int or last_close > cast(int, declaration['start_ns']):
                raise ValueError('Holdout fitted research source overlaps final window or lacks recorded coverage')
        manifest = cast(MLManifest, vars(member)['_manifest'])
        source = source_spec(manifest, binding)
        if source[0] == 'local_file':
            pinned = cast(list[object], declared['source'])
            params = cast(dict[str, object], pinned[1])
            config = manifest.data_source_config
            if config is None:
                raise ValueError('Holdout source declaration is missing')
            config.params['file_path_or_url'] = params['file_path_or_url']
            source = source_spec(manifest, binding)
        identity = {key: value for key, value in source[1].items() if key not in ('start_date_limit', 'end_date_limit', 'row_count_limit', 'n_rows')}
        recorded = cast(dict[str, object], proof['source'])
        if recorded['interval_seconds'] != binding.interval_seconds or (recorded['kind'] != 'caller_frame' and (recorded['kind'] != source[0] or recorded['identity'] != identity)):
            raise ValueError('Holdout source/interval differs from fitted provenance')
        sources.append((source[0], identity, source[2]))
        proofs.append(proof)
    if any(source != sources[0] for source in sources):
        raise ValueError('Holdout members require one common source identity')
    return proofs


def _valid(bar: BarPrediction, timestamp: object) -> None:
    if bar.reason is not None or bar.datetime != timestamp or isinstance(bar.prediction, bool) or not isinstance(bar.prediction, (int, float)) or bar.prediction not in (0, 1):
        raise ValueError(f'Holdout unavailable/alignment/binary prediction: {bar.reason}')
    if bar.probability is not None and (not math.isfinite(bar.probability) or not 0 <= bar.probability <= 1):
        raise ValueError('Holdout probability must be finite in [0,1]')


def _tape(selection: Sensor | Cohort, raw: pl.DataFrame, start: int, failures: _Failures) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    sensors = members(selection)
    logger = logging.getLogger('limen.inference.sensor')
    logger.addHandler(failures)
    try:
        for index, timestamp in enumerate(raw['datetime'].to_list()):
            bar_ns = int(raw['datetime'].slice(index, 1).dt.epoch('ns')[0])
            if bar_ns < start:
                continue
            bars: list[BarPrediction] = []
            retained: list[dict[str, object]] = []
            for ordinal, sensor in enumerate(sensors):
                failures.member_id, failures.bar_ns = f'{sensor.manifest_id}:{sensor.permutation_id}', bar_ns
                with readonly_prediction([sensor]):
                    bar = sensor.predict(raw.head(index + 1))
                if failures.message is not None:
                    raise ValueError(failures.message)
                _valid(bar, timestamp)
                bars.append(bar)
                retained.append({'index': ordinal, 'id': failures.member_id, 'prediction': int(cast(int, bar.prediction)), 'probability': bar.probability, 'reason': bar.reason})
            aggregate = bars[0] if type(selection) is Sensor else _aggregate(cast(Cohort, selection), bars)
            _valid(aggregate, timestamp)
            result.append({'members': retained, 'prediction': int(cast(int, aggregate.prediction)), 'probability': aggregate.probability, 'reason': aggregate.reason})
        return result
    finally:
        logger.removeHandler(failures)


def _execute(final: pl.DataFrame, tape: list[dict[str, object]], declarations: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[list[dict[str, object]]]]:
    columns: dict[str, object] = {name: final[name].to_numpy() for name in ('open', 'high', 'low', 'close')}
    columns.update(predictions=[row['prediction'] for row in tape], price_change=(final['close'] - final['open']).to_numpy())
    summaries: list[dict[str, object]] = []
    ledgers: list[list[dict[str, object]]] = [[] for _ in tape]
    for index, scenario in enumerate(declarations):
        rate = cast(float, scenario['notional_rate'])
        metrics, execution = snapshot_with_execution(columns, fee_bps=cast(float, scenario['fee_bps']), slip_bps=cast(float, scenario['slip_bps']), notional_rate=rate,
            execution_lag_bars=cast(int, scenario['execution_lag_bars']), take_profit_bps=cast(float | None, scenario['take_profit_bps']), stop_loss_bps=cast(float | None, scenario['stop_loss_bps']))
        summaries.append({'index': index, 'declaration': scenario, 'metrics': reports.return_metrics(metrics, execution.net * rate, execution.pos * rate)})
        for offset in range(len(tape)):
            raw = {'position': float(execution.pos[offset]), 'gross': float(execution.gross[offset]), 'net': float(execution.net[offset]), 'cost': float(execution.gross[offset] - execution.net[offset])}
            ledgers[offset].append({'index': index, **{f'raw_{key}': value for key, value in raw.items()}, **{f'deployed_{key}': value * rate for key, value in raw.items()}})
    return summaries, ledgers


def final_holdout_report(frozen: FrozenHoldoutSelection, *, project_root: str | Path,
                         reservation_id: str, start: str | None = None, end: str | None = None) -> dict[str, object]:
    """Allocate durable exposure, then assess a causal tape from an owned fitted copy."""
    if type(frozen) is not FrozenHoldoutSelection:
        raise ValueError('Holdout requires a frozen selection')
    snapshot, frozen_spec, state_sha, spec_sha, lock = frozen_parts(frozen)
    if not lock.acquire(blocking=False):
        raise ValueError('Holdout requires an idle frozen selection')
    try:
        from limen.experiment._holdout_freeze import prediction_dependencies
        try:
            dependencies = prediction_dependencies(snapshot[0])
        except (ValueError, TypeError) as exc:
            raise ValueError('Frozen holdout prediction dependency changed or became unowned') from exc
        if dependencies != frozen_spec.get('prediction_dependencies'):
            raise ValueError('Frozen holdout prediction dependency changed')
        if digest(frozen_spec) != spec_sha or state_digest(snapshot[0]) != state_sha:
            raise ValueError('Frozen holdout specification/state changed')
        selection = owned_copy(snapshot[0])
        target = copy.deepcopy(snapshot[1])
        if target is not None:
            unchanged(target)
        spec = copy.deepcopy(frozen_spec)
        root = history.resolve_root(project_root)
        attempts.migrate(root)
        with history.admission(root):
            declaration = attempts.reservation(root, reservation_id, start, end, cast(str, spec['context_start']))
            proofs = _metadata(selection, root, declaration, spec)
            coverage = attempts.research_coverage(root, declaration)
            planned = {'context_start_ns': history.utc_day(cast(str, spec['context_start'])), 'start_ns': cast(int, declaration['start_ns']), 'end_ns': cast(int, declaration['end_ns'])}
            declarations = cast(list[dict[str, object]], spec['scenarios'])
            scenarios_sha = digest(declarations)
            failures = _Failures()
            for member, proof in zip(members(selection), proofs, strict=True):
                manifest = cast(MLManifest, vars(member)['_manifest'])
                _ = vars(manifest).pop('_holdout_binding', None)
                vars(manifest)['_holdout_assessment_schemas'] = {'input': proof['input_schema'], 'model': proof['model_schema']}
                vars(manifest)['_holdout_assessment_failure'] = failures.schema
            attempt = attempts.allocate(root, declaration, planned, spec=spec_sha, state=state_sha, scenarios=scenarios_sha)
            attempt_id, stage = cast(str, attempt['id']), 'load'
            source: SourceEvidence | None = None
            try:
                interval = cast(int, declaration['interval_seconds'])
                binding = history.ProjectBinding(root, cast(str, declaration['project_id']), cast(int, declaration['revision']), None, interval)
                raw, source = load_admitted_source(cast(MLManifest, vars(members(selection)[0])['_manifest']), binding, lower_ns=planned['context_start_ns'], upper_ns=planned['end_ns'], complete=True)
                stage = 'predict'
                before = state_digest(selection)
                dependencies = prediction_dependencies(selection)
                tape = _tape(selection, raw, planned['start_ns'], failures)
                if state_digest(selection) != before:
                    raise ValueError('Holdout inference changed fitted state')
                if prediction_dependencies(selection) != dependencies:
                    raise ValueError('Holdout inference changed prediction dependencies')
                failures.member_id, failures.bar_ns = None, None
                final = raw.filter(pl.col('datetime').dt.epoch('ns') >= planned['start_ns'])
                stage = 'labels'
                labels: list[int | None] = cast(list[int | None], target.transform(final)[target.target_name].to_list()) if target is not None else [None] * final.height
                stage = 'execute'
                summaries, ledgers = _execute(final, tape, declarations)
                stage = 'artifacts'
                rows, mapping = reports.retained_rows(raw, interval * 1_000_000_000, planned['start_ns'], tape, labels, target is not None, ledgers)
                finished = time.time_ns()
                report: dict[str, object] = {'schema_version': 1, 'attempt_id': attempt_id, 'assessment_index': attempt['assessment_index'], 'state': 'completed',
                    'created_ns': attempt['created_ns'], 'finished_ns': finished, 'project_id': attempt['project_id'], 'reservation': {key: value for key, value in declaration.items() if key != 'project_id'},
                    'spec': spec, 'spec_sha256': spec_sha, 'state_sha256': state_sha, 'scenarios_sha256': scenarios_sha,
                    'source': source, 'source_column_map': mapping, 'source_schema': [{'name': name, 'dtype': str(dtype)} for name, dtype in raw.schema.items()], 'planned': planned, 'prior': attempt['prior'],
                    'exposure': reports.exposure(declaration, cast(dict[str, object], attempt['prior']), proofs, planned['start_ns'], planned['end_ns'], coverage=coverage),
                    'classification': reports.classification(labels, [cast(int, row['prediction']) for row in tape], [cast(float | None, row['probability']) for row in tape]), 'scenarios': summaries}
                files = reports.artifacts(root, report, rows)
                stage = 'finalize'
                attempts.finalize(root, attempt_id, source=source, artifacts=files, finished_ns=finished)
                return report
            except Exception as exc:
                error = {'stage': stage, 'exception': type(exc).__name__, 'message': str(exc), 'member_id': failures.member_id, 'bar_ns': failures.bar_ns}
                try:
                    attempts.finalize(root, attempt_id, source=source, error=error)
                except Exception as finalization_error:
                    logging.getLogger(__name__).exception('Holdout attempt %s finalization failed', attempt_id)
                    raise ValueError(f'Holdout attempt {attempt_id}: {exc}; finalization failed: {finalization_error}') from exc
                raise ValueError(f'Holdout attempt {attempt_id}: {exc}') from exc
    finally:
        lock.release()
