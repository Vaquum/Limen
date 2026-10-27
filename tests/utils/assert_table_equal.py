import pandas as pd
import polars as pl


def assert_table_equal(actual, expected, **kwargs):
    assert isinstance(actual, pl.DataFrame)
    actual = actual.with_columns(pl.col(name).cast(pl.Float64)
                                  for name, dtype in actual.schema.items() if dtype == pl.Null)
    if isinstance(expected, pl.DataFrame):
        expected = expected.to_pandas()
    pd.testing.assert_frame_equal(actual.to_pandas(), expected, **kwargs)
