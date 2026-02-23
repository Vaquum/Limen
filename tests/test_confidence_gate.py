import polars as pl

from limen.data import HistoricalData
from limen.data.utils import split_data_to_prep_output, split_sequential

CONFIDENCE_THRESHOLD = 0.5


def test_confidence_gate() -> None:
    '''Test confidence gate'''

    historical = HistoricalData()
    historical._get_data_for_test(n_rows=200)

    data = (
        historical.data
        .select(['datetime', 'close'])
        .with_row_index('idx')
        .with_columns([
            # target: prediction label
            (pl.col('idx') % 2).cast(pl.Int8).alias('pred_label'),
            # conf: prediction confidence in decimal scale [0,1]
            pl.when((pl.col('idx') % 5) == 0).then(0.2).otherwise(0.8).alias('pred_pct'),
        ])
        .select(['datetime', 'close', 'pred_pct', 'pred_label'])
    )

    split_data = split_sequential(data, (8, 1, 2))
    expected_train = split_data[0]['pred_label'].to_list()
    expected_test_gated = [
        0 if conf < CONFIDENCE_THRESHOLD else pred_label
        for conf, pred_label in zip(
            split_data[2]['pred_pct'].to_list(),
            split_data[2]['pred_label'].to_list(),
            strict=True,
        )
    ]

    data_dict = split_data_to_prep_output(
        split_data=split_data,
        cols=list(data.columns),
        all_datetimes=data['datetime'].to_list(),
        confidence_col='pred_pct',
        confidence_threshold=CONFIDENCE_THRESHOLD,
        gated_target_col='pred_label',
    )

    assert data_dict['y_train'].to_list() == expected_train
    assert data_dict['y_test'].to_list() == expected_test_gated
    assert any(val == 0 for val in data_dict['y_test'].to_list())


if __name__ == '__main__':

    test_confidence_gate()
