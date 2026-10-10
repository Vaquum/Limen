from math import e, isfinite, sqrt
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

__all__ = ['deflated_sharpe_ratio']

_MINIMUM_OBSERVATIONS = 4
_EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe_ratio(
    returns: npt.NDArray[np.float64], *, n_trials: int,
    trial_sharpe_variance: float,
) -> float:
    '''Return the per-bar DSR probability from Bailey and López de Prado (2014).

    https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf, equation 2.
    Sharpe uses sample standard deviation; skew and Pearson kurtosis use
    centered population moments. Four observations and positive return
    variance are required. One trial has a zero selection benchmark.
    '''
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or values.size < _MINIMUM_OBSERVATIONS:
        raise ValueError('DSR requires a one-dimensional track with at least four returns')
    if not np.isfinite(values).all():
        raise ValueError('DSR returns must be finite')
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError('DSR n_trials must be a positive integer')
    if not isfinite(trial_sharpe_variance) or trial_sharpe_variance < 0:
        raise ValueError('DSR trial_sharpe_variance must be finite and nonnegative')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        spread = float(np.std(values, ddof=1))
        if not isfinite(spread) or spread <= 0:
            raise ValueError('DSR returns require positive finite variance')
        mean = float(np.mean(values))
        sharpe = mean / spread
        standardized = (values - mean) / np.std(values)
        skew = float(np.mean(standardized ** 3))
        kurtosis = float(np.mean(standardized ** 4))
        estimator_variance = 1 - skew * sharpe + (kurtosis - 1) * sharpe ** 2 / 4
    if not isfinite(estimator_variance) or estimator_variance <= 0:
        raise ValueError('DSR Sharpe estimator variance must be positive and finite')
    normal = NormalDist()
    benchmark = 0.0
    if n_trials > 1:
        benchmark = sqrt(trial_sharpe_variance) * (
            (1 - _EULER_MASCHERONI) * -normal.inv_cdf(1 / n_trials)
            + _EULER_MASCHERONI * -normal.inv_cdf(1 / (e * n_trials))
        )
    statistic = (sharpe - benchmark) * sqrt(values.size - 1) / sqrt(estimator_variance)
    if not isfinite(statistic):
        raise ValueError('DSR statistic must be finite')
    return normal.cdf(statistic)
