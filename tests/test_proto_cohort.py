import hashlib
import json
import math
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from limen.cli.commands.run import run_experiment
from limen.cohort import Cohort
from limen.cohort.sfc.top_n import select as select_top_n
from limen.data import historical_data
from limen.experiment.trainer import Trainer
from limen.experiment.trainer.sensor import BarPrediction

_COHORT_CONTRACT_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: test_exp
      mode: development
    sfd:
      manifest:
        type: ml
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 3600
        split_dates:
          train_start: "2025-01-01"
          train_end: "2025-01-15"
          val_start: "2025-01-15"
          val_end: "2025-01-18"
          test_start: "2025-01-18"
          test_end: "2025-01-22"
        indicators:
          - func: limen.indicators.roc
            params:
              period: 1
              group: momentum
        target:
          name: quantile_flag
          class: limen.targets.QuantileBinaryTarget
          fit_params:
            source_column: roc_1
            quantile: 0.5
          transform_params:
            shift: -1
        scaler:
          from_params: scaler_type
        reference_architecture: limen.sfd.reference_architecture.logreg_binary
      params:
        scaler_type: [logreg]
        C: [1.0, 0.5]
        random_state: [42]
    uel:
      n_permutations: 2
      search_strategy:
        type: random
      output_format: csv
''')


class _RaisingMember:

    permutation_id = 'pid_raise'

    @property
    def round_params(self):
        return {'model_architecture': 'logreg_binary'}

    def predict(self, _raw_klines):
        raise RuntimeError('member failed during inference')

    def predict_all(self, _raw_klines):
        raise RuntimeError('member failed during inference')


class _BarMember:

    def __init__(self,
                 permutation_id: str,
                 prediction: float | int | None,
                 probability: float | None = None,
                 reason: str | None = None,
                 architecture: str = 'limen.sfd.foundational_sfd.logreg_binary',
                 manifest_id: str | None = None,
                 n_bars: int = 4):

        self.permutation_id = permutation_id
        self.manifest_id = manifest_id
        self._prediction = prediction
        self._probability = probability
        self._reason = reason
        self._n_bars = n_bars
        self._round_params = {'model_architecture': architecture}

    @property
    def round_params(self) -> dict:

        return dict(self._round_params)

    def predict(self, raw_klines) -> BarPrediction:

        return BarPrediction(
            datetime=None,
            prediction=self._prediction,
            probability=self._probability,
            reason=self._reason,
        )

    def predict_all(self, raw_klines) -> list:

        n = len(raw_klines) if hasattr(raw_klines, '__len__') else self._n_bars
        return [
            BarPrediction(datetime=None, prediction=self._prediction,
                          probability=self._probability, reason=self._reason)
            for _ in range(n)
        ]


def _run_real_experiment(experiment_dir: Path,
                         n_permutations: int = 2) -> list[str]:

    experiment_dir = Path(experiment_dir).resolve()
    original = historical_data.HistoricalData.get_spot_klines
    historical_data.HistoricalData.get_spot_klines = staticmethod(_make_yaml_contract_data)
    try:
        with TemporaryDirectory() as tmpdir, patch('click.echo'), patch('click.secho'):
            yaml_path = Path(tmpdir) / 'exp.yaml'
            yaml_text = _minimal_yaml_cli_contract_text(experiment_dir)
            yaml_text = yaml_text.replace('n_permutations: 2', f'n_permutations: {n_permutations}')
            yaml_path.write_text(yaml_text)
            run_experiment(yaml_path)
    finally:
        historical_data.HistoricalData.get_spot_klines = original

    round_ids: list[str] = []
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


def _write_minimal_cohort_artifacts(experiment_dir: Path,
                                    n_rounds: int = 3,
                                    round_ids: list[str] | None = None,
                                    results: pd.DataFrame | None = None) -> list[str]:

    experiment_dir.mkdir(parents=True, exist_ok=True)

    with (experiment_dir / 'metadata.json').open('w') as f:
        json.dump({'sfd_module': 'limen.sfd.foundational_sfd.logreg_binary'}, f)

    ids: list[str] = [f'id_{i}' for i in range(n_rounds)] if round_ids is None else round_ids

    with (experiment_dir / 'round_data.jsonl').open('w') as f:
        for round_id in ids:
            f.write(json.dumps({
                'round_id': round_id,
                '_round_index': ids.index(round_id),
                'round_params': {
                    'model_architecture': 'limen.sfd.foundational_sfd.logreg_binary',
                },
            }) + '\n')

    if results is not None:
        results.to_csv(experiment_dir / 'results.csv', index=False)

    return ids


def _patch_round_architecture(experiment_dir: Path,
                               architecture_by_round_id: dict[str, str]) -> None:

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
                                  permutation_ids: list[str]) -> tuple[list, pl.DataFrame]:

    original = historical_data.HistoricalData.get_spot_klines
    historical_data.HistoricalData.get_spot_klines = staticmethod(_make_yaml_contract_data)
    try:
        trainer = Trainer(experiment_dir)
        sensors = trainer.train(permutation_ids)
    finally:
        historical_data.HistoricalData.get_spot_klines = original

    post_data = _make_post_training_data()
    return sensors, post_data


def _make_yaml_contract_data(kline_size: int = 3600,
                              n_rows: int | None = None,
                              start_date_limit: object = None,
                              end_date_limit: object = None) -> pl.DataFrame:

    _ = kline_size, start_date_limit, end_date_limit
    n = int(n_rows or 500)
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n)]
    close = [100.0 + 0.02 * i + math.sin(i / 7.0) for i in range(n)]

    return pl.DataFrame({
        'datetime': timestamps,
        'open': [value - 0.1 for value in close],
        'high': [value + 0.2 for value in close],
        'low': [value - 0.2 for value in close],
        'close': close,
        'volume': [1000.0 + i for i in range(n)],
    })


def _make_post_training_data(n: int = 50) -> pl.DataFrame:

    # Dates strictly after test_end (2025-01-22) to avoid training window rejection
    timestamps = [datetime(2025, 1, 23) + timedelta(hours=i) for i in range(n)]
    close = [110.0 + 0.02 * i + math.sin(i / 7.0) for i in range(n)]

    return pl.DataFrame({
        'datetime': timestamps,
        'open': [v - 0.1 for v in close],
        'high': [v + 0.2 for v in close],
        'low': [v - 0.2 for v in close],
        'close': close,
        'volume': [1000.0 + i for i in range(n)],
    })


def _minimal_yaml_cli_contract_text(output_path: Path) -> str:

    return _COHORT_CONTRACT_YAML.replace(
        'output_format: csv',
        f'output_format: csv\n  output_path: "{output_path}"',
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

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
            Cohort(experiment_log_path=str(exp_dir), permutation_ids=['abc', 'abc'])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'must be unique' in str(e)


def test_rejects_unknown_permutation_ids():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _run_real_experiment(exp_dir)

        try:
            Cohort(experiment_log_path=str(exp_dir), permutation_ids=['unknown_hash_xyz'])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'Unknown permutation_ids requested' in str(e)


def test_accepts_string_permutation_ids():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=2)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=round_ids)
        assert cohort.permutation_ids == round_ids


def test_rejects_selector_with_explicit_permutation_ids():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir)

        try:
            Cohort(
                experiment_log_path=str(exp_dir),
                permutation_ids=[ids[0]],
                selector='all',
            )
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'permutation_ids or selector' in str(e)


def test_rejects_selector_params_without_selector():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _write_minimal_cohort_artifacts(exp_dir)

        try:
            Cohort(
                experiment_log_path=str(exp_dir),
                selector_params={'column': 'score'},
            )
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'selector_params requires an explicit selector' in str(e)


# ---------------------------------------------------------------------------
# Selector tests
# ---------------------------------------------------------------------------

def test_callable_selector_receives_contract_context():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir)
        results = pd.DataFrame({'id': ids, 'score': [0.1, 0.2, 0.3]})
        (exp_dir / 'results.csv').unlink(missing_ok=True)
        results.to_csv(exp_dir / 'results.csv', index=False)
        seen = {}

        def selector(context):
            seen['has_results'] = 'results' in context
            seen['available'] = context['available_permutation_ids']
            return [ids[2], ids[0]]

        cohort = Cohort(experiment_log_path=str(exp_dir), selector=selector)

        assert seen == {'has_results': True, 'available': sorted(ids)}
        assert cohort.permutation_ids == [ids[2], ids[0]]


def test_selector_context_orders_string_ids():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        _write_minimal_cohort_artifacts(exp_dir, round_ids=['beta', '1abc', 'alpha'])
        seen = {}

        def selector(context):
            seen['available'] = context['available_permutation_ids']
            return ['1abc', 'alpha']

        cohort = Cohort(experiment_log_path=str(exp_dir), selector=selector)

        assert seen['available'] == ['1abc', 'alpha', 'beta']
        assert cohort.available_permutation_ids == ['1abc', 'alpha', 'beta']
        assert cohort.permutation_ids == ['1abc', 'alpha']


def test_named_top_n_selector_uses_results_column():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir)
        results = pd.DataFrame({'id': ids, 'score': [0.1, 0.9, 0.5]})
        results.to_csv(exp_dir / 'results.csv', index=False)

        cohort = Cohort(
            experiment_log_path=str(exp_dir),
            selector='top_n',
            selector_params={'column': 'score', 'n': 2},
        )

        assert cohort.permutation_ids == [ids[1], ids[2]]


def test_builtin_selector_rejects_boolean_ids():

    results = pd.DataFrame({'id': [True], 'score': [1.0]})

    try:
        select_top_n({'results': results}, column='score', n=1)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'boolean permutation id' in str(e)


def test_builtin_selector_coerces_object_ids_to_strings():

    class ExternalId:

        def __str__(self):
            return 'external-1'

    results = pd.DataFrame({'id': [ExternalId()], 'score': [1.0]})

    assert select_top_n({'results': results}, column='score', n=1) == ['external-1']


def test_backtest_pareto_selector_filters_dominated_and_inactive_rows():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir, n_rounds=4)
        results = pd.DataFrame({
            'id': ids,
            'confusion_tp': [5, 5, 5, 0],
            'confusion_fp': [1, 1, 1, 0],
            'backtest_trade_pnl_net_bps_p50': [10.0, 5.0, 1.0, 100.0],
            'backtest_edge_per_signal_bps_p50': [10.0, 12.0, 1.0, 100.0],
            'backtest_return_on_exposure_p50': [10.0, 8.0, 1.0, 100.0],
            'backtest_drawdown_depth_bps_p50': [-100.0, -80.0, -500.0, 0.0],
            'backtest_cvar_95_return_bps': [-50.0, -40.0, -200.0, 0.0],
        })
        results.to_csv(exp_dir / 'results.csv', index=False)

        cohort = Cohort(
            experiment_log_path=str(exp_dir),
            selector='backtest_pareto',
            selector_params={'target_count': 10, 'min_signals': 1},
        )

        # ids[1] dominates ids[0]; ids[2] dominated; ids[3] inactive
        assert cohort.permutation_ids == [ids[1], ids[0]]


def test_diverse_metrics_selector_clamps_cluster_count():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir, n_rounds=4)
        results = pd.DataFrame({
            'id': ids,
            'backtest_trade_pnl_net_bps_p50': [10.0, 20.0, 30.0, 40.0],
            'backtest_edge_per_signal_bps_p50': [2.0, 4.0, 8.0, 16.0],
            'backtest_return_on_exposure_p50': [5.0, 4.0, 3.0, 2.0],
            'backtest_drawdown_depth_bps_p50': [-10.0, -20.0, -30.0, -40.0],
            'backtest_cvar_95_return_bps': [-1.0, -2.0, -3.0, -4.0],
        })
        results.to_csv(exp_dir / 'results.csv', index=False)

        cohort = Cohort(
            experiment_log_path=str(exp_dir),
            selector='diverse_metrics',
            selector_params={'target_count': 2, 'n_clusters': 20},
        )

        assert len(cohort.permutation_ids) == 2
        assert set(cohort.permutation_ids) <= set(ids)


# ---------------------------------------------------------------------------
# Experiment resolution
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def test_rejects_mixed_architecture_selection():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=2)

        _patch_round_architecture(
            exp_dir,
            {round_ids[0]: 'logreg_v1', round_ids[1]: 'tabpfn_v1'},
        )

        try:
            Cohort(experiment_log_path=str(exp_dir), permutation_ids=round_ids)
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'same architecture' in str(e)


def test_sets_probability_mode_for_probability_capable_architecture():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        assert cohort.architecture_id.endswith('logreg_binary')
        assert cohort.supports_probabilities is True
        assert cohort.aggregation_mode == 'probability_weighted'


def test_sets_fallback_mode_for_non_probability_architecture():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        _patch_round_architecture(exp_dir, {round_ids[0]: 'xgboost_regressor'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])

        assert cohort.architecture_id == 'xgboost_regressor'
        assert cohort.supports_probabilities is False
        assert cohort.aggregation_mode == 'majority_vote'


# ---------------------------------------------------------------------------
# set_members validation
# ---------------------------------------------------------------------------

def test_set_members_rejects_count_mismatch():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=2)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=round_ids)
        sensors, _ = _train_real_members_and_input(exp_dir, [round_ids[0]])

        try:
            cohort.set_members(sensors)
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'count must match' in str(e)


def test_set_members_rejects_missing_permutation_id_binding():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])

        class _NoPidMember:

            def predict(self, _data):
                return BarPrediction(datetime=None, prediction=1, probability=None, reason=None)

        try:
            cohort.set_members([_NoPidMember()])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'must expose permutation_id' in str(e)


def test_set_members_rejects_manifest_id_mismatch():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=2)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=round_ids)

        arch = 'limen.sfd.reference_architecture.logreg_binary'
        m0 = _BarMember(round_ids[0], prediction=1, probability=0.7, architecture=arch,
                        manifest_id='sha256:' + 'a' * 64)
        m1 = _BarMember(round_ids[1], prediction=0, probability=0.3, architecture=arch,
                        manifest_id='sha256:' + 'b' * 64)

        try:
            cohort.set_members([m0, m1])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'manifest_id' in str(e)


# ---------------------------------------------------------------------------
# cohort_id and manifest_id
# ---------------------------------------------------------------------------

def test_cohort_id_computed_after_set_members():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        assert cohort.cohort_id is None

        sensors, _ = _train_real_members_and_input(exp_dir, [round_ids[0]])
        cohort.set_members(sensors)

        assert cohort.cohort_id is not None
        assert cohort.cohort_id.startswith('sha256:')
        assert len(cohort.cohort_id) == len('sha256:') + 64


def test_cohort_id_is_stable_across_calls():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        sensors, _ = _train_real_members_and_input(exp_dir, [round_ids[0]])
        cohort.set_members(sensors)

        first = cohort.cohort_id
        # Rebuilding from same experiment and membership yields identical cohort_id
        cohort2 = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        sensors2, _ = _train_real_members_and_input(exp_dir, [round_ids[0]])
        cohort2.set_members(sensors2)

        assert cohort2.cohort_id == first


def test_manifest_id_set_from_metadata_at_construction():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])

        assert cohort.manifest_id is not None
        assert cohort.manifest_id.startswith('sha256:')

        sensors, _ = _train_real_members_and_input(exp_dir, [round_ids[0]])
        cohort.set_members(sensors)

        assert cohort.manifest_id == sensors[0].manifest_id


def test_manifest_id_consistent_across_members():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=2)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=round_ids)
        mid_before = cohort.manifest_id
        assert mid_before is not None

        sensors, _ = _train_real_members_and_input(exp_dir, round_ids)
        cohort.set_members(sensors)

        assert cohort.manifest_id == mid_before
        assert cohort.manifest_id == sensors[0].manifest_id
        assert cohort.manifest_id == sensors[1].manifest_id


def test_yaml_reference_lineage_stripped_from_metadata():
    # Simulate a committed YAML that has a lineage block injected.
    # _write_metadata() must strip it before storing yaml_reference so the
    # manifest_id hash is identical to a plain (pre-commit) YAML.
    original = historical_data.HistoricalData.get_spot_klines
    historical_data.HistoricalData.get_spot_klines = staticmethod(_make_yaml_contract_data)
    try:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exp_dir = root / 'results' / 'test_exp'
            yaml_text = _minimal_yaml_cli_contract_text(exp_dir)
            # Inject a lineage block (as commit_manifest would)
            yaml_with_lineage = yaml_text + dedent('''\
                lineage:
                  id: sha256:aaaa
                  committed_at: "2025-01-01T00:00:00Z"
            ''')
            yaml_path = root / 'exp.yaml'
            yaml_path.write_text(yaml_with_lineage)
            with patch('click.echo'), patch('click.secho'):
                run_experiment(yaml_path)

            with (exp_dir / 'metadata.json').open('r') as f:
                metadata = json.load(f)

            assert 'yaml_reference' in metadata
            assert 'lineage' not in metadata['yaml_reference']

            # manifest_id must equal the hash of the lineage-free canonical form
            data_no_lineage = dict(metadata['yaml_reference'])
            canonical = json.dumps(data_no_lineage, sort_keys=True, default=str)
            expected_id = 'sha256:' + hashlib.sha256(canonical.encode()).hexdigest()
            assert metadata['manifest_id'] == expected_id
    finally:
        historical_data.HistoricalData.get_spot_klines = original


# ---------------------------------------------------------------------------
# predict() — raw klines
# ---------------------------------------------------------------------------

def test_yaml_cli_artifact_cohort_predict_returns_bar_prediction():

    original_get_spot_klines = historical_data.HistoricalData.get_spot_klines
    historical_data.HistoricalData.get_spot_klines = staticmethod(_make_yaml_contract_data)

    try:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / 'exp.yaml'
            exp_dir = root / 'results' / 'test_exp'
            yaml_path.write_text(_minimal_yaml_cli_contract_text(exp_dir))

            assert run_experiment(yaml_path) is True

            with (exp_dir / 'metadata.json').open('r') as f:
                metadata = json.load(f)
            assert isinstance(metadata.get('manifest_id'), str)
            assert metadata['manifest_id'].startswith('sha256:')

            round_ids = sorted(
                json.loads(line)['round_id']
                for line in (exp_dir / 'round_data.jsonl').read_text().splitlines()
                if line.strip()
            )

            trainer = Trainer(exp_dir)
            sensors = trainer.train([round_ids[0]])
            sensor = sensors[0]

            assert isinstance(sensor.permutation_id, str)
            assert sensor.manifest_id == metadata['manifest_id']

            cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
            cohort.set_members(sensors)

            assert cohort.cohort_id is not None
            assert cohort.manifest_id == metadata['manifest_id']

            post_data = _make_post_training_data()
            bar_pred = cohort.predict(post_data)
            sensor_bar_pred = sensor.predict(post_data)

            assert isinstance(bar_pred, BarPrediction)
            assert bar_pred.reason is None
            assert bar_pred.prediction == sensor_bar_pred.prediction
    finally:
        historical_data.HistoricalData.get_spot_klines = original_get_spot_klines


def test_predict_requires_members():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        post_data = _make_post_training_data()

        try:
            cohort.predict(post_data)
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'no bound members' in str(e)


def test_predict_all_requires_members():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        post_data = _make_post_training_data()

        try:
            cohort.predict_all(post_data)
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'no bound members' in str(e)


def test_predict_all_returns_one_entry_per_bar():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        sensors, post_data = _train_real_members_and_input(exp_dir, [round_ids[0]])
        cohort.set_members(sensors)

        results = cohort.predict_all(post_data)

        assert isinstance(results, list)
        sensor_results = sensors[0].predict_all(post_data)
        assert len(results) == len(sensor_results)
        assert all(isinstance(r, BarPrediction) for r in results)


def test_predict_all_valid_bars_match_sensor():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        sensors, post_data = _train_real_members_and_input(exp_dir, [round_ids[0]])
        cohort.set_members(sensors)

        cohort_results = cohort.predict_all(post_data)
        sensor_results = sensors[0].predict_all(post_data)

        valid_cohort = [r for r in cohort_results if r.reason is None]
        valid_sensor = [r for r in sensor_results if r.reason is None]

        assert len(valid_cohort) == len(valid_sensor)
        for c, s in zip(valid_cohort, valid_sensor, strict=True):
            assert c.prediction == s.prediction


def test_predict_all_propagates_worst_reason():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir)
        m0 = _BarMember(ids[0], prediction=None, reason='warm-up')
        m1 = _BarMember(ids[1], prediction=None, reason='inside-training-window')
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[ids[0], ids[1]])
        cohort._members = [m0, m1]

    bars = [m0.predict(None), m1.predict(None)]
    result = cohort._aggregate_bar_predictions(bars)
    assert result.reason == 'inside-training-window'
    assert result.prediction is None


# ---------------------------------------------------------------------------
# Aggregation with stub members
# ---------------------------------------------------------------------------

def test_probability_weighted_predict_aggregates_mean_prob():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir)
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[ids[0], ids[1]])

        m0 = _BarMember(ids[0], prediction=1, probability=0.8)
        m1 = _BarMember(ids[1], prediction=0, probability=0.3)
        cohort.set_members([m0, m1])

        post_data = _make_post_training_data(n=4)
        result = cohort.predict(post_data)

        assert isinstance(result, BarPrediction)
        assert result.reason is None
        assert abs(result.probability - 0.55) < 1e-9   # mean of 0.8 and 0.3
        assert result.prediction == 1                   # 0.55 > 0.5


def test_majority_vote_uses_threshold_for_regressor():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir)
        _patch_round_architecture(exp_dir, {ids[0]: 'xgboost_regressor', ids[1]: 'xgboost_regressor'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[ids[0], ids[1]])

        m0 = _BarMember(ids[0], prediction=0.1, architecture='xgboost_regressor')
        m1 = _BarMember(ids[1], prediction=-0.2, architecture='xgboost_regressor')
        cohort.set_members([m0, m1])

        post_data = _make_post_training_data(n=4)
        result = cohort.predict(post_data)

        # m0: 0.1 > 0 = 1, m1: -0.2 > 0 = 0 → mean=0.5, not > 0.5 → 0
        assert result.prediction == 0
        assert result.probability is None


def test_single_member_majority_vote_returns_binary():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir)
        _patch_round_architecture(exp_dir, {ids[0]: 'xgboost_regressor'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[ids[0]])

        m = _BarMember(ids[0], prediction=1.1, architecture='xgboost_regressor')
        cohort.set_members([m])

        result = cohort.predict(_make_post_training_data(n=4))
        assert result.prediction == 1
        assert result.probability is None


def test_probability_weighted_majority_vote_fallback_returns_none_probability():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        ids = _write_minimal_cohort_artifacts(exp_dir)
        _patch_round_architecture(exp_dir, {ids[0]: 'xgboost_regressor'})
        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[ids[0]])

        m = _BarMember(ids[0], prediction=0.6, architecture='xgboost_regressor')
        cohort.set_members([m])

        result = cohort.predict(_make_post_training_data(n=4))
        assert result.probability is None


def test_member_failure_propagates():

    with TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / 'exp'
        round_ids = _run_real_experiment(exp_dir, n_permutations=1)

        cohort = Cohort(experiment_log_path=str(exp_dir), permutation_ids=[round_ids[0]])
        cohort._members = [_RaisingMember()]
        cohort._members[0].permutation_id = round_ids[0]

        try:
            cohort.predict(_make_post_training_data())
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'member failed during inference' in str(e)


# ---------------------------------------------------------------------------
# Static aggregation helpers
# ---------------------------------------------------------------------------

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
