from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final[Path] = REPO_ROOT / 'governance.yml'
WORKFLOWS_DIR: Final[Path] = REPO_ROOT / '.github/workflows'
RULESET_PATH: Final[Path] = REPO_ROOT / '.github/rulesets/main.json'


def _mapping(value: object, name: str) -> dict[str, object]:
    assert isinstance(value, dict), f'{name} must be a mapping'
    return {str(key): item for key, item in value.items()}


def _config() -> dict[str, object]:
    return _mapping(yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8')), 'governance.yml')


def _section(name: str) -> dict[str, object]:
    return _mapping(_config().get(name), name)


def _str(section: dict[str, object], key: str) -> str:
    value = section.get(key)
    assert isinstance(value, str), f'{key} must be a string'
    return value


def _int(section: dict[str, object], key: str) -> int:
    value = section.get(key)
    assert isinstance(value, int), f'{key} must be an integer'
    return value


def _str_list(section: dict[str, object], key: str) -> list[str]:
    value = section.get(key)
    assert isinstance(value, list), f'{key} must be a list'
    assert all(isinstance(item, str) for item in value), f'{key} must contain strings'
    return [item for item in value if isinstance(item, str)]


def _required_status_contexts() -> list[str]:
    payload = json.loads(RULESET_PATH.read_text(encoding='utf-8'))
    rules = payload['rules']
    assert isinstance(rules, list)
    for rule in rules:
        rule_map = _mapping(rule, 'ruleset rule')
        if rule_map.get('type') != 'required_status_checks':
            continue
        params = _mapping(rule_map.get('parameters'), 'required_status_checks parameters')
        checks = params.get('required_status_checks')
        assert isinstance(checks, list)
        contexts: list[str] = []
        for check in checks:
            check_map = _mapping(check, 'required status check')
            context = check_map.get('context')
            assert isinstance(context, str)
            contexts.append(context)
        return contexts
    raise AssertionError('required_status_checks rule missing from ruleset snapshot')


# This project ships a package, so two workflows must run on more than the
# floor interpreter: packaging builds and audits the distributions, and the
# install matrix proves the built wheel imports on every supported version.
# Every other workflow runs the floor interpreter named in governance.yml.
MULTI_INTERPRETER_WORKFLOWS: Final[frozenset[str]] = frozenset({
    'pr_checks_packaging.yml',
})
SUPPORTED_PYTHONS: Final[frozenset[str]] = frozenset({'3.10', '3.11', '3.12', '3.13'})


def _setup_python_versions() -> dict[str, list[str]]:
    versions: dict[str, list[str]] = {}
    for workflow in sorted(WORKFLOWS_DIR.glob('*.yml')):
        workflow_payload = _mapping(
            yaml.safe_load(workflow.read_text(encoding='utf-8')), workflow.name
        )
        jobs = _mapping(workflow_payload.get('jobs'), f'{workflow.name}.jobs')
        workflow_versions: list[str] = []
        for job_name, job in jobs.items():
            job_map = _mapping(job, f'{workflow.name}.{job_name}')
            steps = job_map.get('steps')
            assert isinstance(steps, list), f'{workflow.name}.{job_name}.steps must be a list'
            for step in steps:
                step_map = _mapping(step, f'{workflow.name}.{job_name}.step')
                uses = step_map.get('uses')
                if not isinstance(uses, str) or not uses.startswith('actions/setup-python@'):
                    continue
                with_config = _mapping(step_map.get('with'), f'{workflow.name}.{job_name}.with')
                version = with_config.get('python-version')
                assert isinstance(version, str), f'{workflow.name} python-version must be quoted'
                workflow_versions.append(version)
        if workflow_versions:
            versions[workflow.name] = workflow_versions
    return versions


def _requirement_pins(package: str) -> list[str]:
    # Workflows install the compiled dev-env.txt, and operators edit
    # dev-env.in; both must carry exactly one identical pin, so a
    # hand-edited compiled set cannot ship an ungoverned tool while the
    # source still reads correctly.
    sources = [
        REPO_ROOT / 'requirements' / 'ci' / 'dev-env.in',
        REPO_ROOT / 'requirements' / 'ci' / 'dev-env.txt',
    ]
    pins = {
        pin
        for source in sources
        for pin in re.findall(
            rf'^{package}==([0-9.]+)\b', source.read_text(encoding='utf-8'), re.MULTILINE
        )
    }
    return sorted(pins)


def test_governance_config_schema_is_minimal() -> None:
    config = _config()

    assert config['schema_version'] == 1
    assert set(config) == {
        'schema_version',
        'runtime',
        'toolchain',
        'review',
        'slice',
        'ruleset',
    }


def test_ruleset_required_checks_match_config() -> None:
    ruleset_config = _section('ruleset')
    ruleset_snapshot = json.loads(RULESET_PATH.read_text(encoding='utf-8'))

    assert ruleset_snapshot['name'] == _str(ruleset_config, 'name')
    assert _required_status_contexts() == _str_list(ruleset_config, 'required_status_checks')


def test_workflow_runtime_and_tooling_match_config() -> None:
    runtime = _section('runtime')
    toolchain = _section('toolchain')
    python_version = _str(runtime, 'python_version')
    ruff_version = _str(toolchain, 'ruff_version')
    pyright_version = _str(toolchain, 'pyright_version')
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))

    assert _setup_python_versions()
    for workflow_name, versions in _setup_python_versions().items():
        if workflow_name in MULTI_INTERPRETER_WORKFLOWS:
            # These deliberately exercise more than the floor interpreter.
            # Matrix expressions are not literal versions; every literal that
            # remains must still be one this project supports.
            literals = [v for v in versions if not v.startswith('${{')]
            assert set(literals) <= SUPPORTED_PYTHONS, (workflow_name, literals)
            continue
        assert versions == [python_version] * len(versions), workflow_name
    assert _requirement_pins('ruff') == [ruff_version]
    assert _requirement_pins('pyright') == [pyright_version]
    # This project bounds the interpreter at both ends: the floor is the
    # version governance.yml names, and the ceiling is the packaging
    # contract's upper bound. Upstream declares a floor only.
    assert pyproject['project']['requires-python'].startswith(f'>={python_version}')
    # The dev extra is declared as a tight bounded range rather than `==`,
    # so a local `pip install -e ".[dev]"` resolves the same version the
    # hash-locked CI set installs. A loose range here is how the local and
    # CI ruff drifted apart before.
    dev = pyproject['project']['optional-dependencies']['dev']
    assert f'ruff>={ruff_version},<' in ' '.join(dev)
    assert f'pyright>={pyright_version},<' in ' '.join(dev)
    assert pyproject['tool']['pyright']['pythonVersion'] == python_version


def test_review_and_slice_settings_match_config() -> None:
    config = _config()
    assert config['review']['approving_authority'] == 'zero-bang'
    assert config['slice']['label'] == 'slice'
    template = REPO_ROOT / config['slice']['issue_template']
    assert template.is_file()

