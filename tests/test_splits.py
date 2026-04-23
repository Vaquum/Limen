import polars as pl

from limen.data.utils.splits import split_data_to_prep_output
from limen.data.utils.splits import split_data_to_rule_based_prep_output
from limen.data.utils.splits import split_sequential


def _make_splits() -> tuple[list[pl.DataFrame], list]:
    df = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(2025, 1, 1),
            end=pl.datetime(2025, 1, 8),
            interval='1d',
            eager=True,
        ),
        'feature': list(range(8)),
        'target': [1, 0] * 4,
    })
    return split_sequential(df, (6, 1, 1)), df['datetime'].to_list()


def test_prep_output_ml_path() -> None:
    splits, all_datetimes = _make_splits()
    cols = ['datetime', 'feature', 'target']

    result = split_data_to_prep_output(splits, cols, all_datetimes)

    assert set(result.keys()) >= {'x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', '_alignment'}
    assert 'feature' in result['x_train'].columns
    assert 'target' not in result['x_train'].columns
    assert result['y_test'].name == 'target'
    assert 'missing_datetimes' in result['_alignment']
    assert result['_alignment']['missing_datetimes'] == []


def test_rule_based_prep_output_returns_full_dataframes() -> None:
    splits, all_datetimes = _make_splits()

    result = split_data_to_rule_based_prep_output(splits, all_datetimes)

    assert set(result.keys()) == {'train', 'val', 'test', '_alignment'}
    for split in ('train', 'val', 'test'):
        assert isinstance(result[split], pl.DataFrame)
        assert 'datetime' not in result[split].columns
        assert 'feature' in result[split].columns
        assert 'target' in result[split].columns
    assert 'missing_datetimes' in result['_alignment']
    assert result['_alignment']['missing_datetimes'] == []
