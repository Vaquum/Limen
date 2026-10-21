from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from limen.yaml.errors import YAMLError

ASSESSMENT_REQUIRED = {'metric', 'direction', 'top_k'}
ASSESSMENT_OPTIONAL = {'folds', 'predict', 'inference_source'}
HOLDOUT_REQUIRED = {'start', 'end', 'interval_seconds', 'context_start', 'scenarios'}
HOLDOUT_OPTIONAL = {'evaluation_target'}


@dataclass(frozen=True)
class Assessment:
    metric: str
    direction: str
    top_k: int
    folds: tuple[int, ...]
    predict: bool
    inference_source: str | None


def _mapping(value: object, name: str, required: set[str], optional: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    source = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in source):
        raise ValueError(f'{name} must have string keys')
    result = {key: item for key, item in source.items() if isinstance(key, str)}
    missing, unknown = required - result.keys(), result.keys() - required - optional
    if missing or unknown:
        raise ValueError(f'{name}: missing {sorted(missing)}, unknown {sorted(unknown)}')
    return result


def assessment_spec(document: Mapping[str, object]) -> Assessment | None:
    if 'assessment' not in document:
        return None
    spec = _mapping(document['assessment'], 'assessment', ASSESSMENT_REQUIRED, ASSESSMENT_OPTIONAL)
    metric, direction, count = spec['metric'], spec['direction'], spec['top_k']
    if not isinstance(metric, str) or not metric.startswith('val_'):
        raise ValueError('assessment.metric must name an explicit recorded validation metric')
    if direction not in ('maximize', 'minimize'):
        raise ValueError('assessment.direction must be maximize or minimize')
    if (not isinstance(count, int) or isinstance(count, bool)) or count < 1:
        raise ValueError('assessment.top_k must be a positive integer')
    raw_folds = spec.get('folds', [])
    if not isinstance(raw_folds, list):
        raise ValueError('assessment.folds must be a list')
    folds = cast(list[object], raw_folds)
    if any(not isinstance(fold, int) or isinstance(fold, bool) or fold < 0 for fold in folds) or len(set(folds)) != len(folds):
        raise ValueError('assessment.folds must contain distinct nonnegative fold numbers')
    predict = spec.get('predict', False)
    if type(predict) is not bool:
        raise ValueError('assessment.predict must be boolean')
    source = spec.get('inference_source')
    if source is not None and (not isinstance(source, str) or not source.strip() or not predict):
        raise ValueError('assessment.inference_source requires prediction and a nonblank file path')
    return Assessment(metric, cast(str, direction), count, tuple(cast(list[int], folds)), predict, source)


def holdout_spec(document: Mapping[str, object]) -> dict[str, object] | None:
    if 'holdout' not in document:
        return None
    spec = _mapping(document['holdout'], 'holdout', HOLDOUT_REQUIRED, HOLDOUT_OPTIONAL)
    from limen.experiment._holdout_history import utc_day
    start, end, context = (utc_day(str(spec[key])) for key in ('start', 'end', 'context_start'))
    interval = spec['interval_seconds']
    if start >= end or context >= start or (not isinstance(interval, int) or isinstance(interval, bool)) or interval <= 0 or 86400 % interval:
        raise ValueError('holdout requires context_start<start<end and a native interval dividing 86400')
    scenarios = spec['scenarios']
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError('holdout.scenarios must be a nonempty list')
    from limen.experiment._holdout_freeze import scenario_declarations
    _ = scenario_declarations(cast(list[Mapping[str, object]], scenarios))
    if 'evaluation_target' in spec:
        target = _mapping(spec['evaluation_target'], 'holdout.evaluation_target', {'class', 'name'}, {'params'})
        if target['class'] not in ('limen.targets.NextBarUpTarget', 'limen.targets.NextBarDownTarget', 'limen.targets.ThresholdBinaryTarget') or not isinstance(target['name'], str) or not target['name']:
            raise ValueError('holdout.evaluation_target requires a supported fixed binary target and name')
        if not isinstance(target.get('params', {}), Mapping):
            raise ValueError('holdout.evaluation_target.params must be a mapping')
    selection = assessment_spec(document)
    if selection is None or selection.predict:
        raise ValueError('holdout requires assessment selection with ordinary prediction disabled')
    return spec


def check_assessment(document: Mapping[str, object], errors: list[YAMLError]) -> None:
    try:
        _ = assessment_spec(document)
        _ = holdout_spec(document)
    except (TypeError, ValueError) as exc:
        errors.append(YAMLError(str(exc), path='assessment/holdout'))
