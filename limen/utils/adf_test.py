from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl


@dataclass
class AdfResult:

    '''Result of an Augmented Dickey-Fuller stationarity test.'''

    stationary: bool
    p_value: float
    test_statistic: float
    critical_values: dict[str, float]


_INCONCLUSIVE = AdfResult(
    stationary=False,
    p_value=1.0,
    test_statistic=0.0,
    critical_values={},
)


def adf_test(series: pl.Series,
             significance_level: float = 0.05) -> AdfResult:

    '''
    Run Augmented Dickey-Fuller test for stationarity.

    Args:
        series (pl.Series): Time series to test
        significance_level (float): Threshold for rejecting non-stationarity

    Returns:
        AdfResult: Structured test result with stationary flag, p-value,
            test statistic, and critical values
    '''

    if not 0 < significance_level < 1:
        raise ValueError(f"adf_test significance_level must be between 0 and 1, got {significance_level}")

    values = series.cast(pl.Float64).drop_nulls().drop_nans().to_numpy()
    if len(values) == 0:
        return _INCONCLUSIVE

    from statsmodels.tsa.stattools import adfuller

    try:
        result = cast(
            tuple[float, float, int, int, dict[str, float], float],
            adfuller(values, autolag='AIC'),
        )
    except (ValueError, np.linalg.LinAlgError):
        return _INCONCLUSIVE

    return AdfResult(
        stationary=result[1] <= significance_level,
        p_value=result[1],
        test_statistic=result[0],
        critical_values=result[4],
    )
