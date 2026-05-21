from pathlib import Path

from click.testing import CliRunner

from limen.cli.main import cli
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


# ---------------------------------------------------------------------------
# list-templates
# ---------------------------------------------------------------------------

def test_cli_list_templates_exits_0() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['list-templates'])
    assert result.exit_code == 0


def test_cli_list_templates_shows_all_template_names() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['list-templates'])
    for name in ('logreg_binary', 'tabpfn_binary', 'xgboost_regressor', 'rule_based'):
        assert name in result.output


def test_cli_list_templates_shows_descriptions() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['list-templates'])
    assert 'Logistic regression' in result.output


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def test_cli_init_creates_file_from_template() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'logreg_binary'])
        assert result.exit_code == 0
        assert Path('my_exp.yaml').exists()


def test_cli_init_sets_metadata_name_to_output_stem() -> None:
    import yaml
    runner = CliRunner()
    with runner.isolated_filesystem():
        runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'logreg_binary'])
        d = yaml.safe_load(Path('my_exp.yaml').read_text())
        assert d['metadata']['name'] == 'my_exp'


def test_cli_init_shows_success_message() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml', '--template', 'logreg_binary'])
        assert '✓' in result.output


def test_cli_init_without_template_lists_templates_and_exits_1() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['init', 'my_exp.yaml'])
        assert result.exit_code == 1
        assert 'logreg_binary' in result.output


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
