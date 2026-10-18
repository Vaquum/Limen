from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from importlib import import_module
from numbers import Real
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder


class _Regressor(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series) -> object: ...
    def score(self, x: pd.DataFrame, y: pd.Series) -> float: ...


class _Splitter(Protocol):
    def split(self, x: pd.DataFrame, *, groups: npt.ArrayLike) -> Iterator[
        tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    ]: ...


make_pipeline = cast(Callable[..., _Regressor], vars(import_module('sklearn.pipeline'))['make_pipeline'])
permutation_importance = cast(Callable[..., Mapping[str, npt.NDArray[np.float64]]],
                              vars(import_module('sklearn.inspection'))['permutation_importance'])

_MIN_CONFIGURATIONS = 20
_MIN_REPEATS = 2


def experiment_parameter_importance(
    self: object,
    metric: str,
    *,
    parameters: Sequence[str],
    random_state: int = 0,
    n_repeats: int = 10,
) -> pd.DataFrame:
    '''Measure held-out surrogate R² loss when each recorded parameter is shuffled.'''
    if isinstance(parameters, (str, bytes)) or not parameters:
        raise ValueError('Parameter importance requires explicit parameter columns')
    columns = list(parameters)
    if (any(type(name) is not str or not name for name in columns)
            or len(set(columns)) != len(columns) or metric in columns):
        raise ValueError('Parameter columns must be unique names excluding the metric')
    if type(random_state) is not int or not 0 <= random_state < 2**32:
        raise ValueError('random_state must be an integer in [0, 2**32)')
    if type(n_repeats) is not int or n_repeats < _MIN_REPEATS:
        raise ValueError('n_repeats must be an integer of at least 2')

    captured = cast(pd.DataFrame, vars(self)['_captured_log'])
    if (type(metric) is not str or not metric or not captured.columns.is_unique
            or any(name not in captured for name in [*columns, metric])):
        raise ValueError('Parameter importance requires existing, unambiguous columns')
    if 'strict_mode_error' in captured:
        captured = captured.loc[captured['strict_mode_error'].isna()
                                | captured['strict_mode_error'].eq('')]
    frame = captured.loc[:, columns].copy()
    categorical: list[str] = []
    for name in columns:
        values = frame[name]
        if values.isna().any():
            raise ValueError(f'Parameter {name!r} contains missing values')
        if all(isinstance(value, (Real, bool, np.bool_)) for value in values):
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f'Parameter {name!r} must be finite')
        elif all(isinstance(value, str) for value in values):
            categorical.append(name)
        else:
            raise ValueError(f'Parameter {name!r} must contain numeric, boolean or text scalars')
    target = captured[metric]
    if (not all(isinstance(value, (Real, bool, np.bool_)) for value in target)
            or not np.isfinite(target.to_numpy(dtype=float)).all()):
        raise ValueError('Successful rounds require a finite numeric metric')

    groups = frame.groupby(columns, sort=False, observed=True).ngroup().to_numpy()
    n_configurations = len(np.unique(groups))
    if n_configurations < _MIN_CONFIGURATIONS:
        raise ValueError('Parameter importance requires at least 20 distinct configurations')
    splitter = cast(_Splitter, GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state))
    train, evaluation = next(splitter.split(frame, groups=groups))
    x_train, x_evaluation = frame.iloc[train], frame.iloc[evaluation]
    y_train, y_evaluation = target.iloc[train], target.iloc[evaluation]
    if y_evaluation.nunique() == 1:
        raise ValueError('Held-out metric must vary for R² parameter importance')
    encoder = ColumnTransformer(
        [('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)],
        remainder='passthrough',
    )
    model = make_pipeline(encoder, RandomForestRegressor(n_estimators=100, n_jobs=1,
                                                        random_state=random_state))
    _ = model.fit(x_train, y_train)
    baseline = float(model.score(x_evaluation, y_evaluation))
    result = permutation_importance(model, x_evaluation, y_evaluation, scoring='r2',
                                    n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    means = np.asarray(result['importances_mean'], dtype=float)
    deviations = np.asarray(result['importances_std'], dtype=float)
    if not np.isfinite(baseline) or not np.isfinite(means).all() or not np.isfinite(deviations).all():
        raise ValueError('Parameter importance produced nonfinite scores')
    return pd.DataFrame({
        'parameter': columns, 'importance_mean': means, 'importance_std': deviations,
        'baseline_score': baseline, 'n_train': len(train), 'n_evaluation': len(evaluation),
        'n_configurations': n_configurations,
    }).sort_values('importance_mean', ascending=False, kind='stable').reset_index(drop=True)
