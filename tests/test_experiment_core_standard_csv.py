import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl

from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.data.utils.splits import split_data_to_prep_output
from limen.data.utils.splits import split_sequential


def _make_standard_csv_test_data() -> pl.DataFrame:
    opens = [100.0] * 16
    closes = [101.0, 100.0, 99.0, 100.0] * 4

    return pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(2025, 1, 1, 0, 0, 0),
            end=pl.datetime(2025, 1, 1, 15, 0, 0),
            interval='1h',
            eager=True,
        ),
        'feature': list(range(16)),
        'open': opens,
        'close': closes,
        'target': [1, 0, 1, 0] * 4,
    })


def _standard_csv_test_prep(
    data: pl.DataFrame,
    round_params: dict | None = None,
) -> dict:
    _ = round_params
    split_data = split_sequential(data, (3, 0, 1))
    return split_data_to_prep_output(
        split_data,
        list(data.columns),
        data['datetime'].to_list(),
    )


def _standard_csv_test_model(data: dict, round_params: dict) -> dict:
    _ = round_params
    preds = data['y_test'].to_list()
    return {
        'accuracy': 1.0,
        '_preds': preds,
    }


def test_standard_run_csv_round_trips_special_string_fields() -> None:
    note = 'alpha, "beta"\nline2'
    label = 'gamma,delta'

    sfd = SimpleNamespace(
        params=lambda: {'marker': [0]},
        prep=_standard_csv_test_prep,
        model=_standard_csv_test_model,
    )

    with TemporaryDirectory() as tmpdir:
        experiment_name = str(Path(tmpdir) / 'standard_csv')

        uel = UniversalExperimentLoop(
            data=_make_standard_csv_test_data(),
            sfd=sfd,
        )
        uel.run(
            experiment_name=experiment_name,
            n_permutations=1,
            prep_each_round=True,
            random_search=False,
            context_params={'note': note, 'label': label},
        )

        csv_path = Path(f'{experiment_name}.csv')
        with csv_path.open(newline='') as f:
            rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert list(rows[0].keys()) == uel.experiment_log.columns
    assert rows[0]['note'] == note
    assert rows[0]['label'] == label
