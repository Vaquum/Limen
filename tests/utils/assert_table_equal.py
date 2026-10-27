import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal as assert_polars_equal


def assert_table_equal(actual, expected, **kwargs):
    assert isinstance(actual, pl.DataFrame)
    if isinstance(expected, pl.DataFrame):
        if 'check_dtype' in kwargs:
            kwargs['check_dtypes'] = kwargs.pop('check_dtype')
        assert_polars_equal(actual, expected, **kwargs)
        return
    actual = actual.with_columns(pl.col(name).cast(pl.Float64)
                                  for name in actual.columns if actual.height and actual[name].null_count() == actual.height)
    expected = expected.copy()
    for column in expected.columns:
        if expected[column].isna().any():
            expected[column] = expected[column].where(expected[column].notna(), np.nan)
    pd.testing.assert_frame_equal(actual.to_pandas(), expected, **kwargs)
