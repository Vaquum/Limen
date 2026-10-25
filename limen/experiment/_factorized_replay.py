from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.experiment._factorized_contract import ReuseContract
from limen.experiment.manifest_core import Manifest
from limen.metrics.rule_based_metrics import rule_based_metrics
from limen.sfd.reference_architecture import _backtest_evaluation
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy


def validate_output(contract: ReuseContract, data: Mapping[str, object], result: Mapping[str, object]) -> None:
    values = np.asarray(result.get('_preds'))
    frame = cast(pl.DataFrame, data.get('x_test', data.get('test')))
    if values.shape != (len(frame),) or values.dtype.kind not in 'bifu' or not np.isfinite(values).all():
        raise ValueError('factorize stage architecture: finite aligned predictions are required')
    if contract.prediction_kind == 'binary' and not np.isin(values, (0, 1)).all():
        raise ValueError('factorize stage architecture: declared binary output contains other values')
    if result.get('_probs') is not None:
        probabilities = np.asarray(result['_probs'])
        if probabilities.shape != values.shape or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError('factorize stage architecture: invalid aligned probabilities')


def replay(manifest: Manifest, contract: ReuseContract, result: Mapping[str, object], data: dict[str, object],
           params: dict[str, object], positions: dict[str, npt.NDArray[np.int64]] | None,
           record_execution: bool) -> dict[str, object]:
    apply_costs = cast(Callable[..., None], vars(Manifest)['_apply_backtest_cost'])
    apply_costs(manifest, data, params)
    scientific = copy.deepcopy({name: value for name, value in result.items() if name != '_model'})
    if contract.replay is not None:
        return contract.replay(scientific, data, params)
    if positions is None:
        predictions = np.asarray(result['_preds'])
        if contract.prediction_kind == 'continuous':
            predictions = (predictions > 0).astype(int)
        compute = cast(Callable[[object, Mapping[str, object]], dict[str, float]], vars(_backtest_evaluation)['compute_backtest'])
        return scientific | compute(predictions, data)
    rule = RuleBasedStrategy()
    cost_args = cast(Callable[..., dict[str, object]], vars(ReferenceModel)['_cost_kwargs'])
    costs = cost_args(rule, data)
    run_split = cast(Callable[..., dict[str, float]], vars(RuleBasedStrategy)['_backtest_split'])
    summaries: dict[str, dict[str, float]] = {}
    for split in ('train', 'val', 'test'):
        if split == 'test' and record_execution:
            costs.update(_record_execution=True, _alignment=data['_alignment'])
        frame = data[split]
        if not isinstance(frame, pl.DataFrame):
            raise ValueError('factorize requires rule-based DataFrame splits')
        summaries[split] = run_split(rule, frame, positions[split], costs)
    summarize = cast(Callable[..., dict[str, object]], rule_based_metrics)
    return scientific | summarize(positions, summaries)
