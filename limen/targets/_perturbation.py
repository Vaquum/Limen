from collections.abc import Mapping
from hashlib import sha256
import math
from typing import Protocol, cast
from numbers import Integral
from ruamel.yaml.scalarbool import ScalarBoolean

import numpy as np
import polars as pl

from limen._parameter_identity import execution_hash

__all__: list[str] = []
NOISE_KEYS = ('horizon_jitter', 'threshold_noise_std', 'barrier_noise_std')


class _Target(Protocol):
    def transform(self, data: pl.DataFrame, **params: object) -> pl.DataFrame: ...


def parameters(horizon: int, thresholds: tuple[float, ...], *, horizon_jitter: int,
               noise_std: float, noise_seed: int, permutation_id: str | None) -> tuple[int, tuple[float, ...]]:
    if not isinstance(cast(object, horizon_jitter), Integral) or isinstance(horizon_jitter, (bool, ScalarBoolean)) or horizon_jitter < 0:
        raise ValueError('horizon_jitter must be a nonnegative integer')
    if not isinstance(cast(object, noise_seed), Integral) or isinstance(noise_seed, (bool, ScalarBoolean)) or noise_seed < 0:
        raise ValueError('noise_seed must be a nonnegative integer')
    if isinstance(noise_std, (bool, ScalarBoolean)) or not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError('noise_std must be finite and nonnegative')
    if not horizon_jitter and noise_std == 0:
        return horizon, thresholds
    if not isinstance(permutation_id, str) or not permutation_id:
        raise ValueError('Target noise requires a permutation_id')
    if not isinstance(cast(object, horizon), Integral) or isinstance(horizon, (bool, ScalarBoolean)) or horizon < 1 or horizon_jitter >= horizon:
        raise ValueError('horizon_jitter must leave every possible horizon positive')
    identity = sha256(f'limen-target-perturbation-v1:{permutation_id}:{noise_seed}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(identity, 'big'))
    actual = horizon + int(rng.integers(-horizon_jitter, horizon_jitter + 1))
    perturbed = tuple(float(value * (1 + rng.normal(0, noise_std))) for value in thresholds)
    if any(not math.isfinite(value) or value <= 0 for value in perturbed):
        raise ValueError('Realized target thresholds must be positive and finite')
    return actual, perturbed


def smooth_binary(data: pl.DataFrame, target: str, smoothing: float, *, training: bool) -> pl.DataFrame:
    if isinstance(smoothing, (bool, ScalarBoolean)) or not math.isfinite(smoothing) or not 0 <= smoothing < 1:
        raise ValueError('label_smoothing must be finite and in [0, 1)')
    if type(training) is not bool:
        raise ValueError('training must be a bool')
    if smoothing == 0 or not training:
        return data
    if any(value not in (0, 1) for value in data[target].drop_nulls().unique()):
        raise ValueError('Label smoothing requires binary labels')
    return data.with_columns((pl.col(target).cast(pl.Float64) * (1 - smoothing) + .5 * smoothing).alias(target))


def apply(instance: object, data: pl.DataFrame, settings: Mapping[str, object],
          round_params: Mapping[str, object], training: bool) -> pl.DataFrame:
    kwargs = dict(settings)
    if getattr(instance, 'requires_permutation_id', False) is True:
        if {'permutation_id', 'training'} & kwargs.keys():
            raise ValueError('Target permutation_id and training are framework operands')
        enabled = any(kwargs.get(key, 0) != 0 for key in NOISE_KEYS)
        identity = (round_params['_permutation_id'] if '_permutation_id' in round_params else execution_hash(round_params)) if enabled else None
        kwargs.update(permutation_id=identity, training=training)
    return cast(_Target, instance).transform(data, **kwargs)


def refuse_reuse(params: Mapping[str, object]) -> None:
    if any(params.get(key, 0) != 0 for key in NOISE_KEYS):
        raise ValueError('factorize refuses permutation-seeded target noise')
