from collections.abc import Callable, Mapping
import logging
import math
from numbers import Integral
from ruamel.yaml.scalarbool import ScalarBoolean
from types import SimpleNamespace
from typing import cast

import polars as pl

from limen._parameter_identity import execution_hash
from limen.log._experiment_parameter_importance import experiment_parameter_importance

__all__: list[str] = []
logger = logging.getLogger(__name__)
_MIN_REPEATS, _MIN_CONFIGURATIONS = 2, 20


class ForestAnalyzer:
    def __init__(self, metric: str, maximize: bool, metrics: Mapping[str, bool] | None,
                 repeats: int, minimum_score: float) -> None:
        super().__init__()
        if metrics is not None and (not metrics or any(not isinstance(cast(object, key), str) or not key or not isinstance(cast(object, value), (bool, ScalarBoolean)) for key, value in metrics.items())):
            raise ValueError('metrics must map metric names to explicit maximize booleans')
        if metrics is not None and metric in metrics and metrics[metric] != maximize:
            raise ValueError('Primary metric direction conflicts with metrics')
        if not isinstance(repeats, Integral) or isinstance(repeats, (bool, ScalarBoolean)) or repeats < _MIN_REPEATS:
            raise ValueError('n_repeats must be an integer of at least 2')
        if isinstance(minimum_score, (bool, ScalarBoolean)) or not math.isfinite(minimum_score) or not 0 < minimum_score <= 1:
            raise ValueError('min_model_score must be in (0, 1]')
        self.metrics = {metric: maximize, **{key: bool(value) for key, value in (metrics or {}).items()}}
        self.repeats, self.minimum_score = int(repeats), minimum_score

    def analyze(self, log: pl.DataFrame, keys: list[str], resolve: Callable[[str, object], tuple[object, ...]], *, threshold: float,
                seed: int, suggested: set[str], minimum_rows: int) -> list[dict[str, object]]:
        successful = log.filter(pl.col('strict_mode_error').is_null() | (pl.col('strict_mode_error') == '')) if 'strict_mode_error' in log.columns else log
        if successful.height < minimum_rows or successful.select(keys).unique().height < _MIN_CONFIGURATIONS:
            return []
        analyses = [experiment_parameter_importance(SimpleNamespace(_captured_log=successful), metric,
                    parameters=keys, random_state=seed, n_repeats=self.repeats) for metric in self.metrics]
        if any(float(result['baseline_score'][0]) < self.minimum_score for result in analyses):
            logger.info('No forest pruning suggestions: held-out surrogate score below %s', self.minimum_score)
            return []
        eligible = set(keys) - suggested
        for result in analyses:
            eligible &= {str(row['parameter']) for row in result.iter_rows(named=True)
                         if max(0., float(row['importance_mean'])) + 2 * float(row['importance_std']) < threshold}
        actions: list[dict[str, object]] = []
        for key in keys:
            if key not in eligible:
                continue
            means = successful.group_by(key, maintain_order=True).agg([pl.col(metric).mean() for metric in self.metrics])
            choices = [means.sort(metric, descending=maximize, maintain_order=True)[key][0] for metric, maximize in self.metrics.items()]
            if len({execution_hash({key: value}) for value in choices}) != 1:
                continue
            native = resolve(key, choices[0])
            if len(native) != 1:
                logger.warning('No forest suggestion for ambiguous logged parameter %s', key)
                continue
            suggested.add(key)
            actions.append({'op': 'keep_is', 'param': key, 'value': native[0], 'action': 'suggest',
                            'reason': f'low held-out forest importance across {list(self.metrics)}; mean + 2 std < {threshold}'})
        return actions


def configure(metric: str, maximize: bool, analyzer: str, metrics: Mapping[str, bool] | None,
              repeats: int, minimum_score: float) -> ForestAnalyzer | None:
    if analyzer not in ('correlation', 'random_forest'):
        raise ValueError('analyzer must be correlation or random_forest')
    if analyzer == 'correlation':
        if metrics is not None:
            raise ValueError('Multiple metrics require analyzer=random_forest')
        return None
    return ForestAnalyzer(metric, maximize, metrics, repeats, minimum_score)
