import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl

from limen.experiment import experiment_core
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

        csv_path = Path(f"{experiment_name}.csv")
        with csv_path.open(newline='') as f:
            rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert list(rows[0].keys()) == uel.experiment_log.columns
    assert rows[0]['note'] == note
    assert rows[0]['label'] == label


def test_standard_run_csv_does_not_append_duplicate_headers_on_rerun() -> None:
    first_note = 'alpha, "beta"\nline2'
    second_note = 'omega, "delta"\nline2'

    sfd = SimpleNamespace(
        params=lambda: {'marker': [0]},
        prep=_standard_csv_test_prep,
        model=_standard_csv_test_model,
    )

    with TemporaryDirectory() as tmpdir:
        experiment_name = str(Path(tmpdir) / 'standard_csv')

        first_uel = UniversalExperimentLoop(
            data=_make_standard_csv_test_data(),
            sfd=sfd,
        )
        first_uel.run(
            experiment_name=experiment_name,
            n_permutations=1,
            prep_each_round=True,
            random_search=False,
            context_params={'note': first_note},
        )

        second_uel = UniversalExperimentLoop(
            data=_make_standard_csv_test_data(),
            sfd=sfd,
        )
        second_uel.run(
            experiment_name=experiment_name,
            n_permutations=1,
            prep_each_round=True,
            random_search=False,
            context_params={'note': second_note},
        )

        csv_path = Path(f"{experiment_name}.csv")
        with csv_path.open(newline='') as f:
            rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]['note'] == first_note
    assert rows[1]['note'] == second_note
    assert rows[1]['id'] != 'id'


def test_standard_run_csv_uses_experiment_dir() -> None:
    first_note = 'alpha'
    second_note = 'omega'

    sfd = SimpleNamespace(
        params=lambda: {'marker': [0]},
        prep=_standard_csv_test_prep,
        model=_standard_csv_test_model,
    )

    cwd = Path.cwd()
    with TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            experiment_dir = Path('output')
            experiment_name = 'experiment_test'

            first_uel = UniversalExperimentLoop(
                data=_make_standard_csv_test_data(),
                sfd=sfd,
                experiment_dir=experiment_dir,
            )
            first_uel.run(
                experiment_name=experiment_name,
                n_permutations=1,
                prep_each_round=True,
                random_search=False,
                context_params={'note': first_note},
            )

            second_uel = UniversalExperimentLoop(
                data=_make_standard_csv_test_data(),
                sfd=sfd,
                experiment_dir=experiment_dir,
            )
            second_uel.run(
                experiment_name=experiment_name,
                n_permutations=1,
                prep_each_round=True,
                random_search=False,
                context_params={'note': second_note},
            )

            csv_path = experiment_dir / f'{experiment_name}.csv'
            with csv_path.open(newline='') as f:
                rows = list(csv.DictReader(f))

            assert csv_path.exists()
            assert not Path(f'{experiment_name}.csv').exists()
        finally:
            os.chdir(cwd)

    assert len(rows) == 2
    assert rows[0]['note'] == first_note
    assert rows[1]['note'] == second_note
    assert rows[1]['id'] != 'id'


def test_standard_run_rechunks_live_log_without_changing_row_output() -> None:
    original_threshold = experiment_core.STANDARD_RUN_LOG_RECHUNK_THRESHOLD
    experiment_core.STANDARD_RUN_LOG_RECHUNK_THRESHOLD = 4

    try:
        sfd = SimpleNamespace(
            params=lambda: {'marker': list(range(10))},
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
                n_permutations=10,
                prep_each_round=True,
                random_search=False,
            )

            csv_path = Path(f"{experiment_name}.csv")
            with csv_path.open(newline='') as f:
                rows = list(csv.DictReader(f))

        assert uel.experiment_log.height == 10
        assert uel.experiment_log.n_chunks() <= 4
        assert uel.experiment_log['id'].to_list() == list(range(10))
        assert uel.experiment_log['marker'].to_list() == list(range(10))
        assert len(rows) == 10
        assert [int(row['id']) for row in rows] == list(range(10))
        assert [int(row['marker']) for row in rows] == list(range(10))
    finally:
        experiment_core.STANDARD_RUN_LOG_RECHUNK_THRESHOLD = original_threshold
