import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.utils.param_space import ParamSpace
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('imports', [
    'from limen.utils.param_space import ParamSpace\nfrom limen.experiment.param_domain import ParamDomain',
    'from limen.experiment.param_domain import ParamDomain\nfrom limen.utils.param_space import ParamSpace',
])
def test_shared_admission_preserves_both_public_import_orders(imports):
    code = imports + "\nassert ParamSpace({'axis': [1, 2]}, 2).n_permutations == 2\nassert ParamDomain({'axis': [1, 2]}).total_combinations == 2\nprint('verified')"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'verified'


@pytest.mark.parametrize('values', [
    [0.0, 0.0], [{'a': 1, 'b': 2}, {'b': 2, 'a': 1}],
    [(1, 2), [1, 2]], [Path('same'), 'same'],
])
@pytest.mark.parametrize('kind', ['domain', 'legacy'])
def test_colliding_choices_refuse_before_constructor_state(values, kind):
    cls = ParamDomain if kind == 'domain' else ParamSpace
    instance = cls.__new__(cls)
    with pytest.raises(ValueError, match='choices 0 and 1 collide under the execution identity'):
        if kind == 'domain':
            instance.__init__({'axis': values})
        else:
            instance.__init__({'axis': values}, 2)
    assert vars(instance) == {}


@pytest.mark.parametrize('values', [[1, 1.0, True], [0.0, -0.0], [None, 'None'], [math.sin, math.cos]])
def test_identity_distinct_controls_keep_values_and_grid_cardinality(values):
    domain = ParamDomain({'axis': values})
    strategy = GridStrategy(domain)
    emitted, identities = [], []
    for row in strategy:
        emitted.append(row['axis'])
        identities.append(strategy.mark_seen(row))
    assert domain.total_combinations == len(values) == len(emitted) == len(set(identities))
    assert [type(value) for value in emitted] == [type(value) for value in values]
    if values == [0.0, -0.0]:
        assert [math.copysign(1, value) for value in emitted] == [1, -1]


class Observer:
    def __init__(self):
        self.calls = []

    def on_domain_changed(self, domain, changed):
        self.calls.append((domain.get_state(), changed))


def test_restored_collision_preserves_domain_version_and_observers():
    domain = ParamDomain({'axis': [1, 2]})
    assert domain.inject_value('axis', 3)
    observer = Observer()
    domain.add_observer(observer)
    before = domain.get_state(), domain.version
    restored = {'axis': [1, 1]}
    original = copy.deepcopy(restored)
    with pytest.raises(ValueError, match='execution identity'):
        domain.set_state(restored)
    assert (domain.get_state(), domain.version) == before
    assert restored == original and not observer.calls


def test_existing_execution_identity_injection_is_a_true_noop():
    value = Path('same')
    domain = ParamDomain({'axis': [value]})
    observer = Observer()
    domain.add_observer(observer)
    assert domain.inject_value('axis', 'same') is False
    assert domain.values_for('axis') == [value] and domain.version == 0 and not observer.calls


@pytest.mark.parametrize('workers', [1, 2])
def test_serial_and_parallel_checkpoint_collisions_refuse_before_more_work(tmp_path, monkeypatch, workers):
    from limen.sfd.reference_architecture import LogRegBinary
    from tests.test_parallel_rounds import make_loop, parallel_config

    config = parallel_config()
    config['uel'].update(n_permutations=4, feedback_interval=1, checkpoint_interval=1)
    config['uel'].pop('pruning_strategies')

    def stop(log, _msq):
        if log.height == 2:
            first._shutdown_requested = True

    first = make_loop(config, tmp_path, callback=stop)
    options = {'n_permutations': 4, 'prep_each_round': True, 'progress_bar': False, 'workers': workers,
        'record_execution': True, 'record_model_outputs': True,
        'recording': dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)}
    first.run(config['metadata']['name'], **options)
    assert first.interrupted and first.experiment_log.height == 2
    path = tmp_path / 'checkpoint.json'
    checkpoint = json.loads(path.read_text())
    value = checkpoint['domain_state']['C'][0]
    checkpoint['domain_state']['C'] = [value, value]
    path.write_text(json.dumps(checkpoint))
    before = {str(path.relative_to(tmp_path)): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in tmp_path.rglob('*') if path.is_file()}
    resumed = make_loop(config, tmp_path)
    observer = Observer()
    resumed._search_strategy.domain.add_observer(observer)
    state = resumed._search_strategy.domain.get_state(), resumed._search_strategy.domain.version

    def forbidden(*args, **kwargs):
        raise AssertionError('Colliding checkpoint reached preparation, fitting or worker launch')

    monkeypatch.setattr(type(resumed.manifest), '_prepare_data_with_fitted', forbidden)
    monkeypatch.setattr(LogRegBinary, 'train', forbidden)
    monkeypatch.setattr('limen.experiment._parallel_rounds.ProcessPoolExecutor', forbidden)
    with pytest.raises(ValueError, match='execution identity'):
        resumed.run(config['metadata']['name'], resume=True, **options)
    assert (resumed._search_strategy.domain.get_state(), resumed._search_strategy.domain.version) == state
    assert not observer.calls
    assert {str(path.relative_to(tmp_path)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in tmp_path.rglob('*') if path.is_file()} == before


def test_actual_duplicate_logreg_domain_refuses_before_all_fresh_cli_work(tmp_path, monkeypatch):
    config, errors = parse(ROOT / 'limen/yaml/templates/logreg_binary.yaml')
    assert not errors and validate(config).valid
    value = config['sfd']['params']['frac_diff_d'][0]
    config['sfd']['params']['frac_diff_d'] = [value, value]
    config['uel']['n_permutations'] = 300
    config['metadata']['mode'] = 'production'
    checked = validate(config)
    assert [error.path for error in checked.errors] == ['sfd.params.frac_diff_d[1]']
    assert 'choices 0 and 1 collide under the execution identity' in checked.errors[0].message
    submitted = tmp_path / 'duplicate.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    monkeypatch.chdir(tmp_path)

    def snapshot():
        return {str(path.relative_to(tmp_path)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in tmp_path.rglob('*') if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError('Duplicate domain reached compilation or source access')

    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', staticmethod(forbidden))
    before = snapshot()
    for command in (['validate'], ['run', '--dry-run'], ['run', '--no-progress-bar'], ['profile'], ['commit']):
        outcome = CliRunner().invoke(cli, [*command, str(submitted)])
        assert outcome.exit_code == 1, outcome.output
        assert '[sfd.params.frac_diff_d[1]]' in outcome.output and snapshot() == before


def test_parsed_float_spellings_collide_at_the_second_choice():
    config, errors = parse(ROOT / 'limen/yaml/templates/logreg_binary.yaml')
    assert not errors
    config['sfd']['params']['C'] = YAML().load('[1.0, 1.]')
    assert [error.path for error in validate(config).errors] == ['sfd.params.C[1]']


def test_all_shipped_seeded_sequences_match_recorded_observations():
    observation = json.loads((ROOT / 'tests/fixtures/search_identity_sequences.json').read_text())
    assert len(observation['records']) == 27
    for record in observation['records']:
        config, errors = parse(ROOT / record['template'])
        assert not errors
        declaration = {'type': record['strategy'], 'seed': record['seed']}
        if record['strategy'] == 'grid':
            declaration['shuffle'] = record['shuffle']
        config['uel']['search_strategy'] = declaration
        strategy = build_search_strategy(config)
        assert strategy.domain.total_combinations == record['domain_cardinality']
        sequence = []
        for _ in range(record['count']):
            row = next(strategy)
            sequence.append({'id': strategy.mark_seen(row),
                'params': {key: value for key, value in row.items() if not key.startswith('_')}})
        digest = hashlib.sha256(json.dumps(sequence, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()
        assert digest == record['sequence_sha256'], (record['template'], record['strategy'])
