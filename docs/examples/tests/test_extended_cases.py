"""Exercise new scientific branches before repeating installed-wheel publication proof."""
import json
import tarfile

from build_bundle import build
from scenario_oracles import verify_analysis, verify_source_formats, verify


def test_recorded_analysis_population_and_every_reducer(python_scenarios, tmp_path):
    from conftest import ROOT
    inputs = ROOT / 'tests/fixtures'
    raw = python_scenarios.hourly(inputs, 1440)
    result = python_scenarios.analyze_candidates(raw, tmp_path)
    verify_analysis(result, inputs)


def test_evaluated_classifier_composition_and_empty_history(python_scenarios, tmp_path):
    from conftest import ROOT
    inputs = ROOT / 'tests/fixtures'
    result = python_scenarios.calibration(inputs, tmp_path)
    recorded = json.loads(json.dumps(result, allow_nan=False))
    verify('python', 'calibration', recorded, tmp_path, inputs)


def test_recorded_local_source_encodings(python_scenarios):
    from conftest import ROOT
    inputs = ROOT / 'tests/fixtures'
    verify_source_formats(python_scenarios.read_source_formats(inputs), inputs)


def test_recorded_multiclass_metrics(python_scenarios):
    from conftest import ROOT
    from helper_oracles import verify_multiclass
    inputs = ROOT / 'tests/fixtures'
    verify_multiclass(python_scenarios.evaluate_three_classes(python_scenarios.hourly(inputs, 1440)), inputs)


def test_native_reducer_recording_and_source_campaigns(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from limen.cli.main import cli
    from conftest import ROOT
    from cli_proof import verify_extended_campaigns
    archive = tmp_path / 'inputs.tar.gz'
    build(ROOT, archive)
    project = tmp_path / 'project'
    with tarfile.open(archive) as bundle:
        bundle.extractall(project, filter='data')
    monkeypatch.chdir(project)
    names = ['analysis', 'parallel', *['reducer_' + name for name in ('sanity', 'saturation', 'focus', 'correlation')],
             *['recording_' + name for name in ('lean', 'full', 'outputs')],
             *['source_' + name for name in ('csv', 'zip', 'arrow')]]
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', *['compiled-research/' + name + '.yaml' for name in names]])
    print(result.output)
    assert result.exit_code == 0, result.exception
    summary = verify_extended_campaigns(project)
    assert summary['analysis']['n_trials'] == 20
