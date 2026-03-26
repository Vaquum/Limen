from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class AdfResult:

    '''Result of an Augmented Dickey-Fuller stationarity test.'''

    stationary: bool
    p_value: float
    test_statistic: float
    critical_values: dict[str, float]


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

    values = series.drop_nulls().drop_nans().to_numpy()
    if len(values) == 0:
        return AdfResult(
            stationary=False,
            p_value=1.0,
            test_statistic=0.0,
            critical_values={},
        )
    from statsmodels.tsa.stattools import adfuller

    try:
        result = adfuller(values, autolag='AIC')
    except (ValueError, np.linalg.LinAlgError):
        return AdfResult(
            stationary=False,
            p_value=1.0,
            test_statistic=0.0,
            critical_values={},
        )

    return AdfResult(
        stationary=result[1] <= significance_level,
        p_value=result[1],
        test_statistic=result[0],
        critical_values=result[4],
    )
