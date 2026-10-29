import json
from pathlib import Path
import re

import polars as pl
import pytest

from check_evidence import bindings, region
from recorded_inputs import funding_history, target_label_inputs, trade_points

ROOT = Path(__file__).resolve().parents[3]


def test_compiled_project_is_yaml_and_cli():
    project = ROOT / 'docs/examples/compiled-research'
    assert not list(project.rglob('*.py'))
    scripts = '\n'.join(path.read_text() for path in project.rglob('*.sh'))
    assert not re.search(r'\bpython[\d.]*\b|CompiledSFD|UniversalExperimentLoop', scripts)
    for path in project.rglob('*.yaml'):
        assert str(path.relative_to(ROOT / 'docs/examples')) in scripts, path
    commands = json.loads((ROOT / 'docs/examples/commands.json').read_text())
    for name, command in commands.items():
        if name.startswith('compiled-'):
            assert command.startswith(('limen ', 'sh compiled-research/'))


def test_compiled_python_substitution_fails_even_when_excerpt_matches(tmp_path):
    path = tmp_path / 'substituted.md'
    path.write_text('<!-- example: substituted source=docs/examples/recorded_inputs.py region=trade_profile scenario=bars interface=compiled role=executed -->\n```python\nignored\n```\n')
    with pytest.raises(ValueError, match='requires YAML and CLI'):
        bindings(ROOT, path)


@pytest.mark.parametrize('name', ['features', 'run'])
def test_research_yaml_cannot_claim_the_selection_command(name, tmp_path):
    path = tmp_path / 'wrong-scenario.md'
    source = 'docs/examples/compiled-research/research.yaml'
    path.write_text(f'<!-- example: wrong-scenario source={source} region={name} scenario=selection interface=compiled role=executed -->\n```yaml\n' + region(ROOT / source, name) + '```\n')
    with pytest.raises(ValueError, match='not consumed by its Compiled scenario'):
        bindings(ROOT, path)


def test_bundled_derivations_equal_the_recorded_sources():
    inputs = ROOT / 'tests/fixtures'
    assert pl.read_parquet(inputs / 'spot_1h_20240101_20240116.parquet').equals(pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet').head(384))
    assert pl.read_parquet(inputs / 'recorded-funding.parquet').equals(funding_history(inputs))
    assert pl.read_parquet(inputs / 'recorded-trade-points.parquet').equals(trade_points(inputs))
    for name, expected in target_label_inputs(inputs).items():
        assert pl.read_parquet(inputs / ('recorded-' + name + '-labels.parquet')).equals(expected)
