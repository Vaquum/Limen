from collections.abc import Mapping
import numbers
from typing import cast

import numpy as np
import numpy.typing as npt

__all__ = ['snapshot_signals']


def snapshot_signals(
    predictions: object, *, class_mapping: Mapping[int | str, int] | None = None,
    label_index: int | None = None,
) -> npt.NDArray[np.int64]:
    if class_mapping is not None and label_index is not None:
        raise ValueError('snapshot class_mapping and label_index are mutually exclusive')
    try:
        values = np.asarray(cast(npt.ArrayLike, predictions), dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError('snapshot predictions must be a rectangular array') from exc
    if label_index is not None:
        if isinstance(label_index, bool) or not isinstance(label_index, numbers.Integral) or label_index < 0:
            raise ValueError('snapshot label_index must be a nonnegative integer')
        if values.ndim != 2 or not values.size:
            raise ValueError('snapshot label_index requires a nonempty 2D binary indicator matrix')
        if label_index >= values.shape[1]:
            raise ValueError('snapshot label_index is outside the indicator matrix')
        if not np.isin(values, (0, 1)).all():
            raise ValueError('snapshot multilabel predictions must contain only binary indicators')
        return values[:, label_index].astype(np.int64)
    if not isinstance(class_mapping, Mapping) or not class_mapping:
        raise ValueError('snapshot class_mapping must be a nonempty mapping')
    if any(isinstance(key, bool) or not isinstance(key, (str, numbers.Integral)) for key in class_mapping):
        raise ValueError('snapshot class_mapping keys must be integer or string class identities')
    if any(type(value) is not int or value not in (0, 1) for value in class_mapping.values()):
        raise ValueError('snapshot class_mapping values must be literal binary 0 or 1')
    if values.ndim != 1 or not values.size:
        raise ValueError('snapshot class_mapping requires nonempty 1D class identities')
    decisions: list[int] = []
    for identity in cast(list[object], values.tolist()):
        if isinstance(identity, bool) or not isinstance(identity, (str, numbers.Integral)):
            raise ValueError('snapshot classes must be nonmissing integer or string identities')
        key = identity if isinstance(identity, str) else int(identity)
        if key not in class_mapping:
            raise ValueError(f'snapshot class_mapping has no decision for class {key!r}')
        decisions.append(class_mapping[key])
    return np.asarray(decisions, dtype=np.int64)
