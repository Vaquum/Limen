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
