from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import pickle
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.backtest.trade_contract import contract_digest, json_value
from limen.calibration._objective_threshold import validate_probabilities
from limen.experiment._allocation import AllocationBatch
from limen.experiment._prepare_trade_context import PreparedTradeContext

if TYPE_CHECKING:
    from limen.experiment._objective import ValidationScorer

__all__: list[str] = []


def validation_inputs(data: Mapping[str, object], scorer: ValidationScorer) -> dict[str, object]:
    batches = data.get('_allocation_batches')
    if isinstance(batches, Mapping):
        batch = batches.get('val')
        if not isinstance(batch, AllocationBatch) or (batch.partition_start_ns, batch.partition_end_ns) != (scorer.inputs.partition_start_ns, scorer.inputs.partition_end_ns) or batch.available_at_ns != tuple(scorer.inputs.signals['available_at_ns'].to_list()) or len(batch.row_ids) != scorer.inputs.signals.height:
            raise ValueError('Objective allocation validation batch does not match its outer partition')
        _ = batch.digest()
        return {'allocation_batch': deepcopy(batch)}
    if 'x_val' in data:
        return {'x_test': deepcopy(data['x_val'])}
    frame, strategy = cast(pl.DataFrame, data['val']), cast(Mapping[str, object], data['strategy'])
    predicates = [str(item['id']) for item in cast(list[Mapping[str, object]], strategy['conditions']) if 'type' in item]
    return {'test': frame.clone(), 'strategy': deepcopy(strategy), 'x_test': frame.select(predicates).to_numpy()}


def validation_header(data: Mapping[str, object], scorer: ValidationScorer, inputs: Mapping[str, object]) -> dict[str, object]:
    original = cast(PreparedTradeContext, data['_trade_context']).partitions[1]
    checksum = sha256(pickle.dumps((inputs, data.get('y_val')), protocol=5)).hexdigest()
    binding = contract_digest({'partition': [scorer.inputs.partition_start_ns, scorer.inputs.partition_end_ns],
                               'original_sources': json_value(original.sources), 'sources': json_value(scorer.inputs.sources),
                               'policy': json_value(scorer.policy), 'prepared': checksum,
                               'rows': json_value(scorer.inputs.signals.select('row_id', 'available_at_ns').to_dicts())})
    return {'row_ids': scorer.inputs.signals['row_id'].to_list(), 'source_binding': binding, 'prediction_mode': scorer.policy.prediction_mode}


def _vector(value: object, rows: int) -> npt.NDArray[np.float64]:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.kind not in 'bifu' or not np.isfinite(array).all():
        raise ValueError('Objective requires finite ordered validation decisions matching its rows')
    return array.astype(np.float64)


def validation_evidence(inputs: Mapping[str, object], result: Mapping[str, object], scorer: ValidationScorer,
                        header: Mapping[str, object]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None, Mapping[str, object]]:
    explicit = '_validation' in result
    model = result.get('_model')
    if explicit:
        payload = result['_validation']
        if not isinstance(payload, Mapping):
            raise ValueError('Objective validation evidence must be a mapping')
        payload = cast(Mapping[str, object], payload)
        if any(payload.get(key) != value for key, value in header.items()):
            raise ValueError('Objective validation evidence has stale source, row order or prediction mode')
        prediction, probability = payload.get('predictions'), payload.get('probabilities')
    else:
        predict = getattr(model, 'predict', None)
        if not callable(predict):
            raise ValueError('Objective requires explicit validation evidence or a fitted predictor')
        first, second = predict(deepcopy(dict(inputs))), predict(deepcopy(dict(inputs)))
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise ValueError('Objective fitted prediction must return a mapping')
        for key in ('_preds', '_probs'):
            if (first.get(key) is None) != (second.get(key) is None) or (first.get(key) is not None and not np.array_equal(first[key], second[key])):
                raise ValueError('Objective validation predictions must be stable and ordered')
        prediction, probability, payload = first.get('_preds'), first.get('_probs'), {}
    modes = (result.get('_prediction_mode'), getattr(model, 'prediction_mode', None), payload.get('prediction_mode'))
    if any(mode is not None and mode != scorer.policy.prediction_mode for mode in modes) or not any(mode is not None for mode in modes):
        raise ValueError('Objective prediction mode must match the declared execution policy')
    predictions = _vector(prediction, scorer.inputs.signals.height)
    probabilities = None if probability is None else validate_probabilities(probability)
    if probabilities is not None and probabilities.size != predictions.size:
        raise ValueError('Objective probabilities do not match validation decisions')
    return predictions, probabilities, payload
