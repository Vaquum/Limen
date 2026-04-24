import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from limen.cohort import Cohort
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.experiment.trainer import Trainer
from limen.sfd.foundational_sfd import logreg_binary as logreg_sfd


class _RaisingMember:

    permutation_id = 0

    def predict(self, _data):
        raise RuntimeError('member failed during inference')


class _FallbackContinuousMember:

    def __init__(self,
                 permutation_id: int,
                 preds: np.ndarray,
                 *,
                 architecture: str = 'xgboost_regressor'):

        self.permutation_id = permutation_id
        self._preds = np.asarray(preds, dtype=float)
        self._round_params = {'model_architecture': architecture}
        self._metadata = {
            'sfd_module': 'limen.sfd.foundational_sfd.logreg_binary'}

    @property
    def round_params(self) -> dict:

        return dict(self._round_params)

    @property
    def metadata(self) -> dict:

        return dict(self._metadata)

    def predict(self, _data):

        return {'_preds': self._preds.copy()}


def _run_real_experiment(experiment_dir: Path,
                         n_permutations: int = 2) -> list[int]:

    params = logreg_sfd.params()
    domain = ParamDomain(params)
    strategy = GridStrategy(domain)

    uel = UniversalExperimentLoop(
        sfd=logreg_sfd,
        search_strategy=strategy,
        experiment_dir=experiment_dir,
    )

    uel.run(
        experiment_name='test_proto_cohort',
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


def _train_real_members_and_input(experiment_dir: Path,
                                  permutation_ids: list[int]) -> tuple[list, np.ndarray]:

    trainer = Trainer(experiment_dir)
    sensors = trainer.train(permutation_ids)

    data_dict = trainer._manifest.prepare_data(
        trainer._data, sensors[0].round_params)
    x_test = data_dict['x_test']

    return sensors, x_test


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

    missing_path = Path.cwd() / 'does-not-exist-limen-123456'

    try:
        Cohort(experiment_log_path=str(missing_path))
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


def test_rejects_raw_input_for_sensor_compatible_predict_contract():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])
        sensors, _x_test = _train_real_members_and_input(exp_dir, [0])
        cohort.set_members(sensors)

        try:
            cohort.predict(np.array([[1.0, 2.0]], dtype=float))
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'decoder-style dict input' in str(e)


def test_rejects_raw_input_for_dict_required_architecture():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        _patch_round_architecture(exp_dir, {0: 'tabpfn_binary'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])

        class _StubMember:

            permutation_id = 0

            def predict(self, _data):
                return {'_preds': np.array([1], dtype=np.int8), '_probs': np.array([0.9], dtype=float)}

        cohort.set_members([_StubMember()])

        try:
            cohort.predict(np.array([[1.0, 2.0]], dtype=float))
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'decoder-style dict input' in str(e)


def test_accepts_dict_input_for_dict_required_architecture():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        _patch_round_architecture(exp_dir, {0: 'tabpfn_binary'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])

        class _CaptureMember:

            permutation_id = 0

            def __init__(self):
                self.last_input = None

            def predict(self, data):
                self.last_input = data
                return {'_preds': np.array([1], dtype=np.int8), '_probs': np.array([0.9], dtype=float)}

        member = _CaptureMember()
        cohort.set_members([member])

        data = {
            'x_test': np.array([[1.0, 2.0]], dtype=float),
            'x_val': np.array([[0.5, 1.5]], dtype=float),
            'y_val': np.array([1], dtype=np.int8),
        }
        out = cohort.predict(data)

        assert out['_preds'].tolist() == [1]
        assert member.last_input == data


def test_set_members_rejects_count_mismatch():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=2)

        cohort = Cohort(experiment_log_path=str(
            exp_dir), permutation_ids=[0, 1])
        sensors, _x_test = _train_real_members_and_input(exp_dir, [0])

        try:
            cohort.set_members(sensors)
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'count must match' in str(e)


def test_set_members_rejects_missing_permutation_id_binding():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])

        class _NoPidMember:

            def predict(self, _data):
                return {'_preds': np.array([1], dtype=np.int8)}

        try:
            cohort.set_members([_NoPidMember()])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'must expose permutation_id' in str(e)


def test_member_specific_payload_is_routed_by_permutation_id():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=2)

        sensors, x_test = _train_real_members_and_input(exp_dir, [0, 1])
        cohort = Cohort(experiment_log_path=str(
            exp_dir), permutation_ids=[0, 1])
        cohort.set_members(sensors)

        routed = {
            0: {'x_test': x_test[:3]},
            1: {'x_test': x_test},
        }

        try:
            cohort.predict({'x_test': x_test, '_by_permutation_id': routed})
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'same shape' in str(e)


def test_sets_fallback_mode_for_non_probability_architecture():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        _patch_round_architecture(exp_dir, {0: 'xgboost_regressor'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])

        assert cohort.architecture_id == 'xgboost_regressor'
        assert cohort.supports_probabilities is False
        assert cohort.aggregation_mode == 'majority_vote'


def test_probability_weighted_predict_aggregates_mean_p1():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=2)

        sensors, x_test = _train_real_members_and_input(exp_dir, [0, 1])

        cohort = Cohort(experiment_log_path=str(
            exp_dir), permutation_ids=[0, 1])
        cohort.set_members(sensors)

        y_pred = cohort.predict({'x_test': x_test})['_preds']

        member_probs = [
            np.asarray(sensor.predict({'x_test': x_test})[
                       '_probs'], dtype=float)
            for sensor in sensors
        ]
        expected = (np.mean(np.vstack(member_probs), axis=0)
                    > 0.5).astype(np.int8)

        assert np.array_equal(y_pred, expected)


def test_probability_weighted_predict_single_member_matches_own_thresholded_probs():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        sensors, x_test = _train_real_members_and_input(exp_dir, [0])

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])
        cohort.set_members(sensors)

        y_pred = cohort.predict({'x_test': x_test})['_preds']
        expected = np.asarray(sensors[0].predict({'x_test': x_test})['_preds'])

        assert np.array_equal(y_pred, expected)


def test_single_decoder_passthrough_returns_member_preds_unchanged():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        sensors, x_test = _train_real_members_and_input(exp_dir, [0])

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])
        cohort.set_members(sensors)

        y_pred = cohort.predict({'x_test': x_test})['_preds']
        expected = np.asarray(sensors[0].predict({'x_test': x_test})['_preds'])

        assert np.array_equal(y_pred, expected)


def test_probability_weighted_predict_requires_members():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])

        try:
            cohort.predict([[1]])
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'no bound decoder members' in str(e)


def test_majority_vote_uses_binary_votes_for_continuous_fallback_members():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=2)

        _patch_round_architecture(
            exp_dir,
            {
                0: 'xgboost_regressor',
                1: 'xgboost_regressor',
            },
        )
        cohort = Cohort(experiment_log_path=str(
            exp_dir), permutation_ids=[0, 1])

        m0 = _FallbackContinuousMember(
            0,
            np.array([0.1, -0.2, 0.8, -0.4], dtype=float),
        )
        m1 = _FallbackContinuousMember(
            1,
            np.array([0.2, -0.7, 0.6, -0.3], dtype=float),
        )
        cohort.set_members([m0, m1])

        out = cohort.predict({'x_test': np.zeros((4, 2), dtype=float)})

        # Directional fallback votes use threshold > 0 for regressor outputs.
        # m0=[1,0,1,0], m1=[1,0,1,0] => [1,0,1,0]
        assert out['_preds'].tolist() == [1, 0, 1, 0]


def test_majority_vote_tie_returns_zero():

    vote = Cohort._majority_vote([
        np.array([1, 0, 1, 0], dtype=float),
        np.array([0, 1, 1, 0], dtype=float),
    ])

    assert vote.tolist() == [0, 0, 1, 0]


def test_probability_weighted_tie_returns_zero():

    vote = Cohort._probability_weighted_vote([
        np.array([0.6, 0.4, 0.5], dtype=float),
        np.array([0.4, 0.6, 0.5], dtype=float),
    ])

    assert vote.tolist() == [0, 0, 0]


def test_majority_vote_multimember_expected_output():

    vote = Cohort._majority_vote([
        np.array([1, 1, 0, 0], dtype=float),
        np.array([1, 0, 1, 0], dtype=float),
        np.array([1, 0, 0, 1], dtype=float),
    ])

    assert vote.tolist() == [1, 0, 0, 0]


def test_probability_weighted_vote_rejects_shape_mismatch():

    try:
        Cohort._probability_weighted_vote([
            np.array([0.6, 0.4], dtype=float),
            np.array([0.4], dtype=float),
        ])
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'same shape' in str(e)


def test_majority_vote_rejects_shape_mismatch():

    try:
        Cohort._majority_vote([
            np.array([1, 0], dtype=float),
            np.array([1], dtype=float),
        ])
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'same shape' in str(e)


def test_predict_return_probs_probability_mode_returns_per_sample_probs():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=2)

        sensors, x_test = _train_real_members_and_input(exp_dir, [0, 1])

        cohort = Cohort(experiment_log_path=str(
            exp_dir), permutation_ids=[0, 1])
        cohort.set_members(sensors)

        y_pred, probs = cohort.predict({'x_test': x_test}, return_probs=True)

        assert isinstance(probs, np.ndarray)
        assert np.asarray(probs).shape == (2, np.asarray(y_pred).shape[0])


def test_predict_return_meta_returns_metadata_placeholder():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        sensors, x_test = _train_real_members_and_input(exp_dir, [0])

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])
        cohort.set_members(sensors)

        y_pred, meta = cohort.predict({'x_test': x_test}, return_meta=True)

        assert np.asarray(y_pred).shape[0] == np.asarray(x_test).shape[0]
        assert meta['permutation_ids'] == [0]
        assert meta['decoder_count'] == 1
        assert 'architecture_id' in meta
        assert 'aggregation_mode' in meta


def test_predict_return_probs_and_return_meta_returns_three_tuple():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=2)

        sensors, x_test = _train_real_members_and_input(exp_dir, [0, 1])

        cohort = Cohort(experiment_log_path=str(
            exp_dir), permutation_ids=[0, 1])
        cohort.set_members(sensors)

        y_pred, probs, meta = cohort.predict(
            {'x_test': x_test},
            return_probs=True,
            return_meta=True,
        )

        assert isinstance(probs, np.ndarray)
        assert np.asarray(probs).shape == (2, np.asarray(y_pred).shape[0])
        assert meta['permutation_ids'] == [0, 1]
        assert meta['decoder_count'] == 2
        assert np.asarray(y_pred).shape[0] == np.asarray(x_test).shape[0]


def test_predict_return_probs_rejected_in_fallback_mode():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        _patch_round_architecture(exp_dir, {0: 'xgboost_regressor'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])

        member = _FallbackContinuousMember(
            0,
            np.array([0.9, 0.1], dtype=float),
        )
        cohort.set_members([member])

        try:
            cohort.predict(
                {'x_test': np.zeros((2, 2), dtype=float)}, return_probs=True)
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'Probabilities are unavailable' in str(e)


def test_validate_probability_range_accepts_values_in_unit_interval():

    Cohort._validate_probability_range(np.array([0.0, 0.25, 0.5, 1.0]))


def test_validate_probability_range_rejects_out_of_range_values():

    try:
        Cohort._validate_probability_range(np.array([0.2, 1.1, -0.1]))
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'within [0, 1]' in str(e)


def test_validate_probability_range_rejects_non_finite_values():

    try:
        Cohort._validate_probability_range(np.array([0.2, np.nan, 0.8]))
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'finite values' in str(e)


def test_cohort_is_drop_in_decoder_replacement_for_dict_input():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        sensors, x_test = _train_real_members_and_input(exp_dir, [0])
        base_sensor = sensors[0]

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])
        cohort.set_members([base_sensor])

        live_data = {'x_test': x_test}

        sensor_result = base_sensor(live_data)
        cohort_result = cohort(live_data)
        cohort_pred = cohort.predict(live_data)

        assert isinstance(sensor_result, dict)
        assert isinstance(cohort_result, dict)
        assert isinstance(cohort_pred, dict)
        assert '_preds' in cohort_result
        assert '_probs' in cohort_result
        assert np.array_equal(np.asarray(cohort_result['_preds']),
                              np.asarray(sensor_result['_preds']))
        assert np.allclose(np.asarray(cohort_result['_probs'], dtype=float),
                           np.asarray(sensor_result['_probs'], dtype=float))
        assert np.array_equal(np.asarray(cohort_pred['_preds']),
                              np.asarray(sensor_result['_preds']))


def test_single_member_probability_mode_preserves_exact_payload_shape_and_keys():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])

        class _NonTrivialProbMember:

            permutation_id = 0

            def predict(self, _data):
                return {
                    '_preds': np.array([0, 1], dtype=np.int8),
                    '_probs': np.array([0.9, 0.1], dtype=float),
                    'optimal_threshold': 0.9,
                    'val_score': 0.123,
                }

        member = _NonTrivialProbMember()
        cohort.set_members([member])

        out = cohort.predict({'x_test': np.zeros((2, 2), dtype=float)})

        expected = member.predict({'x_test': np.zeros((2, 2), dtype=float)})
        assert set(out.keys()) == set(expected.keys())
        assert np.array_equal(out['_preds'], expected['_preds'])
        assert np.array_equal(out['_probs'], expected['_probs'])
        assert out['optimal_threshold'] == expected['optimal_threshold']
        assert out['val_score'] == expected['val_score']


def test_member_failure_propagates_and_fails_whole_call():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir, n_permutations=1)

        _sensors, x_test = _train_real_members_and_input(exp_dir, [0])

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[0])
        cohort.set_members([_RaisingMember()])

        try:
            cohort.predict({'x_test': x_test})
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'member failed during inference' in str(e)
