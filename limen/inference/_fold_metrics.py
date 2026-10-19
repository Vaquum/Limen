from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import cast

import numpy as np

from limen.experiment._fold_evidence import model_witness
from limen.experiment._recording import project_record
from limen.sfd.reference_architecture.base import ReferenceModel

__all__: list[str] = []
_ABSOLUTE_TOLERANCE = 1e-6


def scalar_matches(expected: object, actual: object, deterministic: bool) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, Real) and isinstance(actual, Real):
        left, right = float(expected), float(actual)
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        if not math.isfinite(left) or not math.isfinite(right):
            return left == right
        difference = abs(left - right)
        return difference <= _ABSOLUTE_TOLERANCE or (not deterministic and difference <= 0.01 * max(abs(left), abs(right), 1.0))
    return expected == actual


def validate(expected: Mapping[str, object], actual: Mapping[str, object], deterministic: bool) -> None:
    mismatches = [key for key, value in expected.items() if key not in actual or not scalar_matches(value, actual[key], deterministic)]
    if mismatches:
        raise ValueError('selected fold metric mismatch: ' + ', '.join(mismatches))


def bindings(expected: Mapping[str, object], data: Mapping[str, object], witness: Mapping[str, object], model: ReferenceModel, policy: Mapping[str, bool]) -> None:
    for key in ('trade_contract', 'trade_contract_digest'):
        if project_record(expected, policy).get(key) != project_record(data, policy).get(key):
            raise ValueError(f'exact selected fold {key} changed')
    if expected.get('learning_binding') != data.get('_learning_binding'):
        raise ValueError('exact selected fold learning_binding changed')
    threshold = getattr(model, '_calibration_threshold', None)
    if not scalar_matches(witness['fitted_threshold'], threshold, True):
        raise ValueError('selected fold fitted threshold changed')
    if model_witness(model) != witness['inference_model']:
        raise ValueError('selected fold inference model/output contract changed')


def stable_model(model: ReferenceModel, data: Mapping[str, object]) -> None:
    inputs = np.concatenate([np.asarray(data['x_' + name], dtype=float) for name in ('train', 'val', 'test')])
    if not len(inputs):
        raise ValueError('No retained inputs for native inference stability validation')
    batch = model.predict({'x_test': inputs})
    preds = np.asarray(batch.get('_preds'), dtype=float).reshape(-1)
    if len(preds) != len(inputs) or not np.isfinite(preds).all():
        raise ValueError('Native inference requires one finite scalar prediction per retained input')
    declaration = cast(Mapping[str, object] | None, model_witness(model)['output_contract'])
    if declaration is not None:
        if declaration['kind'] == 'binary' and not np.isin(preds, [0, 1]).all():
            raise ValueError('Declared binary inference returned a nonbinary prediction')
        if declaration['probabilities']:
            probs = np.asarray(batch.get('_probs'), dtype=float).reshape(-1)
            if len(probs) != len(inputs) or not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
                raise ValueError('Declared inference probability is absent or outside [0, 1]')
    indices = sorted({0, len(inputs) // 2, len(inputs) - 1})
    sample = inputs[indices]
    first, repeated = model.predict({'x_test': sample}), model.predict({'x_test': sample})
    rows = [model.predict({'x_test': row.reshape(1, -1)}) for row in sample]
    for key in ('_preds', '_probs'):
        if first.get(key) is None and repeated.get(key) is None and all(row.get(key) is None for row in rows):
            continue
        values = np.asarray(first.get(key), dtype=float).reshape(-1)
        singles = np.concatenate([np.asarray(row.get(key), dtype=float).reshape(-1) for row in rows])
        if not np.array_equal(values, np.asarray(repeated.get(key)).reshape(-1), equal_nan=True) or not np.allclose(values, singles, rtol=0, atol=1e-12) or not np.allclose(values, np.asarray(batch[key])[indices], rtol=0, atol=1e-12):
            raise ValueError('Native inference is not stable across repeated, row and batch predictions')
