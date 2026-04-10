import json
from pathlib import Path
from tempfile import TemporaryDirectory

from limen.cohort import Cohort
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.sfd.foundational_sfd import logreg_binary as logreg_sfd
from tests.stubs.stubs import StubStrategy


def _run_real_experiment(experiment_dir: Path,
                         n_permutations: int = 2) -> list[int]:

    params = logreg_sfd.params()
    domain = ParamDomain(params)
    strategy = StubStrategy(domain)

    uel = UniversalExperimentLoop(
        sfd=logreg_sfd,
        search_strategy=strategy,
        experiment_dir=experiment_dir,
    )

    uel.run(
        experiment_name='test_cohort_constructor',
        n_permutations=n_permutations,
    )

    round_ids: list[int] = []
    with (experiment_dir / 'round_data.jsonl').open('r') as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            round_ids.append(json.loads(stripped)['round_id'])

    return sorted(round_ids)


def _write_real_metadata_only(experiment_dir: Path) -> None:

    experiment_dir.mkdir(parents=True, exist_ok=True)

    with (experiment_dir / 'metadata.json').open('w') as f:
        json.dump({'sfd_module': 'limen.sfd.foundational_sfd.logreg_binary'}, f)


def _patch_round_architecture(experiment_dir: Path,
                              architecture_by_round_id: dict[int, str]) -> None:

    round_data_path = experiment_dir / 'round_data.jsonl'
    rows: list[dict] = []

    with round_data_path.open('r') as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue

            entry = json.loads(stripped)
            rid = entry.get('round_id')
            if rid in architecture_by_round_id:
                rp = dict(entry.get('round_params', {}))
                rp['model_architecture'] = architecture_by_round_id[rid]
                entry['round_params'] = rp
            rows.append(entry)

    with round_data_path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def test_rejects_when_no_source_provided():

    try:
        Cohort()
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'exactly one' in str(e)


def test_rejects_when_both_sources_provided():

    try:
        Cohort(experiment_id='x', experiment_log_path='y')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'exactly one experiment source' in str(e)


def test_rejects_missing_experiment_log_path():

    try:
        Cohort(experiment_log_path='/tmp/does-not-exist-limen-123456')
        assert False, 'Expected FileNotFoundError'
    except FileNotFoundError as e:
        assert 'missing or unreadable' in str(e)


def test_defaults_to_all_permutations_when_not_provided():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        expected_ids = _run_real_experiment(exp_dir, n_permutations=3)

        cohort = Cohort(experiment_log_path=str(exp_dir))
        assert cohort.available_permutation_ids == expected_ids
        assert cohort.permutation_ids == expected_ids


def test_rejects_empty_permutation_ids():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir)

        try:
            Cohort(experiment_log_path=str(exp_dir), permutation_ids=[])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'non-empty list' in str(e)


def test_rejects_duplicate_permutation_ids():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir)

        try:
            Cohort(experiment_log_path=str(exp_dir), permutation_ids=[1, '1'])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'must be unique' in str(e)


def test_rejects_unknown_permutation_ids():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir)

        try:
            Cohort(experiment_log_path=str(exp_dir), permutation_ids=[99])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'Unknown permutation_ids requested' in str(e)


def test_accepts_string_permutation_ids_when_numeric():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir)

        cohort = Cohort(experiment_log_path=str(
            exp_dir), permutation_ids=['0', '1'])
        assert cohort.permutation_ids == [0, 1]


def test_rejects_when_round_data_is_missing():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _write_real_metadata_only(exp_dir)

        try:
            Cohort(experiment_log_path=str(exp_dir))
            assert False, 'Expected FileNotFoundError'
        except FileNotFoundError as e:
            assert 'missing or unreadable' in str(e)


def test_rejects_unresolvable_experiment_id():

    try:
        Cohort(experiment_id='nonexistent-experiment-id-xyz')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Unable to resolve experiment_id' in str(e)


def test_rejects_ambiguous_experiment_id_resolution():

    with TemporaryDirectory(dir='.') as tmp1, TemporaryDirectory(dir='.') as tmp2:
        exp_name = 'dup-exp-id'

        exp_dir_1 = Path(tmp1) / exp_name
        exp_dir_2 = Path(tmp2) / exp_name
        _run_real_experiment(exp_dir_1, n_permutations=1)
        _run_real_experiment(exp_dir_2, n_permutations=1)

        try:
            Cohort(experiment_id=exp_name)
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'resolved to multiple experiment logs' in str(e)


def test_rejects_mixed_architecture_selection():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=2)

        _patch_round_architecture(
            exp_dir,
            {
                0: 'logreg_v1',
                1: 'tabpfn_v1',
            },
        )

        try:
            Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0, 1])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'same architecture' in str(e)


def test_sets_probability_mode_for_probability_capable_architecture():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])
        assert cohort.architecture_id.endswith('logreg_binary')
        assert cohort.supports_probabilities is True
        assert cohort.aggregation_mode == 'probability_weighted'


def test_sets_fallback_mode_for_non_probability_architecture():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        _patch_round_architecture(exp_dir, {0: 'xgboost_regressor'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])

        assert cohort.architecture_id == 'xgboost_regressor'
        assert cohort.supports_probabilities is False
        assert cohort.aggregation_mode == 'majority_vote'
