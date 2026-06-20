import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from limen.cli.main import cli
from limen.yaml.parser import parse
from limen.yaml.profiler import ProfileResult
from tests.test_yaml import _MINIMAL_ML_YAML


def test_cli_validate_valid_yaml_exits_0() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        result = runner.invoke(cli, ['validate', 'exp.yaml'])
        assert result.exit_code == 0


def test_cli_validate_shows_valid_checkmark() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        result = runner.invoke(cli, ['validate', 'exp.yaml'])
        assert '✓ Valid' in result.output


def test_cli_validate_parse_error_exits_1() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text('sfd: [unclosed')
        result = runner.invoke(cli, ['validate', 'exp.yaml'])
        assert result.exit_code == 1


def test_cli_validate_schema_error_exits_1_and_shows_error() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML.replace('name: test_exp\n', ''))
        result = runner.invoke(cli, ['validate', 'exp.yaml'])
        assert result.exit_code == 1
        assert 'ERROR' in result.output


def test_cli_run_dry_run_valid_yaml_exits_0() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        result = runner.invoke(cli, ['run', '--dry-run', 'exp.yaml'])
        assert result.exit_code == 0


def test_cli_run_dry_run_shows_dry_run_message() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        result = runner.invoke(cli, ['run', '--dry-run', 'exp.yaml'])
        assert 'Dry run' in result.output


def test_cli_run_dry_run_parse_error_exits_1() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text('sfd: [unclosed')
        result = runner.invoke(cli, ['run', '--dry-run', 'exp.yaml'])
        assert result.exit_code == 1


def test_cli_run_dry_run_valid_yaml_shows_dry_run_complete() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        result = runner.invoke(cli, ['run', '--dry-run', 'exp.yaml'])
        assert result.exit_code == 0
        assert 'Dry run' in result.output


def test_cli_list_templates_exits_0() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['list-templates'])
    assert result.exit_code == 0


def test_cli_list_templates_shows_all_template_names() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['list-templates'])
    for name in ('lightgbm_binary', 'logreg_binary', 'tabpfn_binary', 'xgboost_regressor', 'rule_based'):
        assert name in result.output


def test_cli_list_templates_shows_descriptions() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['list-templates'])
    assert 'Logistic regression' in result.output


def test_cli_init_creates_file_from_template() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'logreg_binary'])
        assert result.exit_code == 0
        assert Path('my_exp.yaml').exists()


def test_cli_init_sets_metadata_name_to_output_stem() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'logreg_binary'])
        d, _ = parse(Path('my_exp.yaml'))
        assert d['metadata']['name'] == 'my_exp'


def test_cli_init_shows_success_message() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'logreg_binary'])
        assert '✓' in result.output


def test_cli_init_without_template_prompts_and_uses_default() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml'], input='\n')
        assert result.exit_code == 0
        assert 'logreg_binary' in result.output
        assert Path('my_exp.yaml').exists()


def test_cli_init_without_template_accepts_typed_name() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml'], input='rule_based\n')
        assert result.exit_code == 0
        d, _ = parse(Path('my_exp.yaml'))
        assert d['metadata']['name'] == 'my_exp'


def test_cli_init_bare_name_appends_yaml_extension() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp', '--template', 'logreg_binary'])
        assert result.exit_code == 0
        assert Path('my_exp.yaml').exists()


def test_cli_init_bare_name_inside_project_places_in_manifests() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('limen.toml').write_text('[project]\nname = "test"\n')
        result = runner.invoke(cli, ['init', 'my_exp', '--template', 'logreg_binary'])
        assert result.exit_code == 0
        assert Path('manifests/my_exp.yaml').exists()
        assert 'manifests/my_exp.yaml' in result.output


def test_cli_init_bare_name_outside_project_warns_and_creates_in_cwd() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp', '--template', 'logreg_binary'])
        assert result.exit_code == 0
        assert Path('my_exp.yaml').exists()
        assert 'no limen.toml' in result.output


def test_cli_init_unknown_template_exits_1() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'nonexistent'])
        assert result.exit_code == 1
        assert 'not found' in result.output


def test_cli_init_refuses_to_overwrite_existing_file() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('my_exp.yaml').write_text('existing content')
        result = runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'logreg_binary'])
        assert result.exit_code == 1
        assert 'already exists' in result.output
        assert Path('my_exp.yaml').read_text() == 'existing content'


def test_cli_init_rejects_invalid_slug_stem() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my exp.yaml', '--template', 'logreg_binary'])
        assert result.exit_code == 1
        assert 'my exp' in result.output


def test_cli_init_metadata_name_not_updated_in_non_metadata_fields() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'rule_based'])
        assert result.exit_code == 0
        d, _ = parse(Path('my_exp.yaml'))
        assert d['metadata']['name'] == 'my_exp'
        text = Path('my_exp.yaml').read_text()
        assert 'name: "RSI oversold"' in text or 'name: RSI oversold' in text



def _make_results_dir(yaml_reference: dict | None = None,
                      target_permutations: int = 10) -> tempfile.TemporaryDirectory:
    td = tempfile.TemporaryDirectory()
    tmp = Path(td.name)
    if yaml_reference is not None:
        metadata = {'sfd_module': 'yaml:test_exp', 'yaml_reference': yaml_reference}
        (tmp / 'metadata.json').write_text(json.dumps(metadata))
    checkpoint = {
        'metadata': {
            'experiment_round': 5,
            'target_permutations': target_permutations,
            'strategy_type': 'random',
            'content_hash': 'abc',
        },
        'msq_state': {},
        'domain_state': {},
    }
    (tmp / 'checkpoint.json').write_text(json.dumps(checkpoint))
    return td


def test_cli_run_resume_exits_0_on_success() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    with _make_results_dir(yaml_reference=dict(yaml_dict)) as _tmpdir:
        runner = CliRunner()
        with patch('limen.cli.commands.resume.UniversalExperimentLoop') as mock_uel:
            mock_uel.return_value.run.return_value = None
            result = runner.invoke(cli, ['run', '--resume', _tmpdir])
        assert result.exit_code == 0


def test_cli_run_resume_calls_uel_run_with_resume_true() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    with _make_results_dir(yaml_reference=dict(yaml_dict), target_permutations=20) as _tmpdir:
        runner = CliRunner()
        with patch('limen.cli.commands.resume.UniversalExperimentLoop') as mock_uel:
            mock_uel.return_value.run.return_value = None
            runner.invoke(cli, ['run', '--resume', _tmpdir])
        _, kwargs = mock_uel.return_value.run.call_args
    assert kwargs.get('resume') is True
    assert kwargs.get('n_permutations') == 20


def test_cli_run_resume_errors_when_no_metadata_json() -> None:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)
        checkpoint = {'metadata': {'experiment_round': 1, 'target_permutations': 10,
                                   'strategy_type': 'random', 'content_hash': 'x'},
                      'msq_state': {}, 'domain_state': {}}
        (tmp / 'checkpoint.json').write_text(json.dumps(checkpoint))
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--resume', str(tmp)])
        assert result.exit_code == 1
        assert 'metadata.json' in result.output


def test_cli_run_resume_errors_when_no_yaml_reference() -> None:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)
        (tmp / 'metadata.json').write_text(json.dumps({'sfd_module': 'some.sfd'}))
        checkpoint = {'metadata': {'experiment_round': 1, 'target_permutations': 10,
                                   'strategy_type': 'random', 'content_hash': 'x'},
                      'msq_state': {}, 'domain_state': {}}
        (tmp / 'checkpoint.json').write_text(json.dumps(checkpoint))
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--resume', str(tmp)])
        assert result.exit_code == 1
        assert 'yaml_reference' in result.output


def test_cli_run_resume_errors_when_no_checkpoint_json() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)
        (tmp / 'metadata.json').write_text(json.dumps({'yaml_reference': dict(yaml_dict)}))
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--resume', str(tmp)])
        assert result.exit_code == 1
        assert 'checkpoint' in result.output.lower()


def test_cli_run_resume_and_yaml_file_together_exits_1() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    with _make_results_dir(yaml_reference=dict(yaml_dict)) as _tmpdir:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
            result = runner.invoke(cli, ['run', 'exp.yaml', '--resume', _tmpdir])
        assert result.exit_code == 1
        assert 'Cannot specify both' in result.output


def test_cli_run_no_args_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['run'])
    assert result.exit_code == 1


def test_cli_run_dry_run_and_resume_together_exits_1() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    with _make_results_dir(yaml_reference=dict(yaml_dict)) as _tmpdir:
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--dry-run', '--resume', _tmpdir])
        assert result.exit_code == 1
        assert 'dry-run' in result.output


def test_cli_run_resume_errors_when_checkpoint_missing_metadata_key() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)
        (tmp / 'metadata.json').write_text(json.dumps({'yaml_reference': dict(yaml_dict)}))
        (tmp / 'checkpoint.json').write_text(json.dumps({'msq_state': {}, 'domain_state': {}}))
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--resume', str(tmp)])
        assert result.exit_code == 1
        assert 'checkpoint' in result.output.lower()


def test_cli_run_resume_errors_when_metadata_json_is_invalid_json() -> None:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)
        (tmp / 'metadata.json').write_text('not valid json {{{')
        checkpoint = {'metadata': {'experiment_round': 1, 'target_permutations': 10,
                                   'strategy_type': 'random', 'content_hash': 'x'},
                      'msq_state': {}, 'domain_state': {}}
        (tmp / 'checkpoint.json').write_text(json.dumps(checkpoint))
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--resume', str(tmp)])
        assert result.exit_code == 1
        assert 'metadata.json' in result.output


def test_cli_run_resume_errors_when_yaml_reference_is_not_a_dict() -> None:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)
        (tmp / 'metadata.json').write_text(json.dumps({'yaml_reference': 'not-a-dict'}))
        checkpoint = {'metadata': {'experiment_round': 1, 'target_permutations': 10,
                                   'strategy_type': 'random', 'content_hash': 'x'},
                      'msq_state': {}, 'domain_state': {}}
        (tmp / 'checkpoint.json').write_text(json.dumps(checkpoint))
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--resume', str(tmp)])
        assert result.exit_code == 1
        assert 'yaml_reference' in result.output


def _make_profile_result(*, attempted: int = 0, completed: int = 0,
                         time_per: float | None = None,
                         warnings: list | None = None,
                         errors: list | None = None) -> ProfileResult:
    return ProfileResult(
        total_permutations=120,
        param_cardinalities={'solver': 3, 'C': 4, 'max_iter': 2, 'tol': 5},
        complexity_rating='low',
        sample_permutations_attempted=attempted,
        sample_permutations_completed=completed,
        sample_time_seconds_per_permutation=time_per,
        warnings=warnings or [],
        errors=errors or [],
    )


def test_cli_profile_valid_yaml_exits_0() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        with patch('limen.cli.commands.profile.profile', return_value=_make_profile_result()):
            result = runner.invoke(cli, ['profile', 'exp.yaml'])
        assert result.exit_code == 0


def test_cli_profile_shows_permutation_count_and_rating() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        with patch('limen.cli.commands.profile.profile', return_value=_make_profile_result()):
            result = runner.invoke(cli, ['profile', 'exp.yaml'])
        assert '120' in result.output
        assert 'low' in result.output


def test_cli_profile_shows_all_param_names() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        with patch('limen.cli.commands.profile.profile', return_value=_make_profile_result()):
            result = runner.invoke(cli, ['profile', 'exp.yaml'])
        for name in ('solver', 'C', 'max_iter', 'tol'):
            assert name in result.output


def test_cli_profile_shows_runtime_skipped_when_no_sampling() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        prof = _make_profile_result(
            warnings=['No permutations completed — timing unavailable.'],
        )
        with patch('limen.cli.commands.profile.profile', return_value=prof):
            result = runner.invoke(cli, ['profile', 'exp.yaml'])
        assert 'Skipped' in result.output


def test_cli_profile_shows_timing_when_sampling_completed() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        prof = _make_profile_result(attempted=5, completed=5, time_per=0.42)
        with patch('limen.cli.commands.profile.profile', return_value=prof):
            result = runner.invoke(cli, ['profile', 'exp.yaml'])
        assert '0.420s' in result.output
        assert '5 of 5' in result.output


def test_cli_profile_shows_errors_from_sampling() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML)
        prof = _make_profile_result(
            attempted=3, completed=1, time_per=0.1,
            errors=['Data too small for profiling — extend the date range in data_source.'],
        )
        with patch('limen.cli.commands.profile.profile', return_value=prof):
            result = runner.invoke(cli, ['profile', 'exp.yaml'])
        assert 'ERROR' in result.output
        assert 'date range' in result.output


def test_cli_profile_parse_error_exits_1() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text('sfd: [unclosed')
        result = runner.invoke(cli, ['profile', 'exp.yaml'])
        assert result.exit_code == 1


def test_cli_profile_validation_error_exits_1() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path('exp.yaml').write_text(_MINIMAL_ML_YAML.replace('name: test_exp\n', ''))
        result = runner.invoke(cli, ['profile', 'exp.yaml'])
        assert result.exit_code == 1
        assert 'ERROR' in result.output


def test_cli_run_resume_errors_when_yaml_reference_missing_metadata_name() -> None:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)
        (tmp / 'metadata.json').write_text(json.dumps({'yaml_reference': {'sfd': {}}}))
        checkpoint = {'metadata': {'experiment_round': 1, 'target_permutations': 10,
                                   'strategy_type': 'random', 'content_hash': 'x'},
                      'msq_state': {}, 'domain_state': {}}
        (tmp / 'checkpoint.json').write_text(json.dumps(checkpoint))
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--resume', str(tmp)])
        assert result.exit_code == 1
        assert 'metadata.name' in result.output
