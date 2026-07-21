import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl
import pytest

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


def test_standard_run_rejects_existing_csv_without_header() -> None:
    sfd = SimpleNamespace(
        params=lambda: {'marker': [0]},
        prep=_standard_csv_test_prep,
        model=_standard_csv_test_model,
    )

    with TemporaryDirectory() as tmpdir:
        experiment_name = str(Path(tmpdir) / 'standard_csv')
        Path(f'{experiment_name}.csv').write_text('\n')

        uel = UniversalExperimentLoop(
            data=_make_standard_csv_test_data(),
            sfd=sfd,
        )

        with pytest.raises(ValueError, match='Existing results CSV has no header'):
            uel.run(
                experiment_name=experiment_name,
                n_permutations=1,
                prep_each_round=True,
                random_search=False,
            )


def test_standard_run_batches_live_log_without_changing_row_output() -> None:
    original_batch_size = experiment_core.STANDARD_RUN_LOG_BATCH_SIZE
    experiment_core.STANDARD_RUN_LOG_BATCH_SIZE = 4

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
        experiment_core.STANDARD_RUN_LOG_BATCH_SIZE = original_batch_size


class _StandardCsvFakeManifest:
    data_source_config = object()
    test_data_source_config = None

    def __init__(self) -> None:
        self.prepare_calls = 0

    def architecture_function(self, data: dict, **kw: dict) -> dict:
        return {}

    def prepare_data(self, data: pl.DataFrame, round_params: dict) -> dict:
        self.prepare_calls += 1
        return _standard_csv_test_prep(data, round_params)

    def run_model(self, data: dict, round_params: dict) -> dict:
        return _standard_csv_test_model(data, round_params)


def _make_standard_csv_manifest_sfd() -> SimpleNamespace:
    manifest = _StandardCsvFakeManifest()
    return SimpleNamespace(
        params=lambda: {'marker': [0, 1]},
        manifest=lambda: manifest,
    )


def test_standard_run_manifest_default_auto_resolves_prep_each_round() -> None:
    sfd = _make_standard_csv_manifest_sfd()
    manifest = sfd.manifest()

    with TemporaryDirectory() as tmpdir:
        experiment_name = str(Path(tmpdir) / 'standard_csv')

        uel = UniversalExperimentLoop(
            data=_make_standard_csv_test_data(),
            sfd=sfd,
        )
        uel.run(
            experiment_name=experiment_name,
            n_permutations=2,
            random_search=False,
        )

    assert manifest.prepare_calls == 2
    assert uel.experiment_log.height == 2


def test_standard_run_manifest_explicit_prep_each_round_false_raises() -> None:
    sfd = _make_standard_csv_manifest_sfd()

    with TemporaryDirectory() as tmpdir:
        experiment_name = str(Path(tmpdir) / 'standard_csv')

        uel = UniversalExperimentLoop(
            data=_make_standard_csv_test_data(),
            sfd=sfd,
        )

        with pytest.raises(ValueError, match='prep_each_round must be True for manifest-driven SFDs'):
            uel.run(
                experiment_name=experiment_name,
                n_permutations=1,
                prep_each_round=False,
                random_search=False,
            )


def test_standard_run_custom_sfd_default_auto_resolves_prep_first_round_only() -> None:
    prep_calls = []

    def _counting_prep(data: pl.DataFrame, round_params: dict | None = None) -> dict:
        prep_calls.append(True)
        return _standard_csv_test_prep(data, round_params)

    sfd = SimpleNamespace(
        params=lambda: {'marker': [0, 1]},
        prep=_counting_prep,
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
            n_permutations=2,
            random_search=False,
        )

    assert len(prep_calls) == 1
    assert uel.experiment_log.height == 2


def test_standard_run_skips_post_processing_by_default() -> None:
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
        finalize_calls = []
        uel.extras = [{'stale': True}]

        def fake_finalize():
            finalize_calls.append(True)

        uel._finalize = fake_finalize
        uel.run(
            experiment_name=experiment_name,
            n_permutations=1,
            prep_each_round=True,
            random_search=False,
        )

    assert finalize_calls == []
    assert uel.experiment_log.height == 1
    assert uel.round_params == []
    assert uel.preds == []
    assert uel.scalers == []
    assert uel._alignment == []
    assert uel.extras == []
    assert uel._log is None
    assert uel.experiment_confusion_metrics is None
    assert uel.experiment_backtest_results is None
    assert uel.experiment_parameter_correlation is None
