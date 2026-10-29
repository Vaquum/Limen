from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import shlex
import signal
import tarfile
import venv
import time
import zipfile

from build_bundle import INPUTS, build, build_results
from prepare_environment import verify_wheelhouse
from released_wheel import verify_download
from scenario_oracles import verify


def run(argv, *, cwd, env=None, timeout: float | None = 90):
    environment = dict(os.environ if env is None else env)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        environment.pop(name, None)
    environment['PYTHONNOUSERSITE'] = '1'
    started = time.monotonic()
    command = [str(arg) for arg in argv]
    with subprocess.Popen(command, cwd=cwd, env=environment, start_new_session=True) as process:
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            raise
        if status:
            raise subprocess.CalledProcessError(status, command)
    print(f'Wheel proof stage {time.monotonic() - started:.2f}s: ' + ' '.join(map(str, argv)), flush=True)


def archive_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError('Missing regular archive file: ' + name)
    with stream:
        return stream.read()


def unpack(bundle: Path, target: Path, root: Path):
    with tarfile.open(bundle) as archive:
        names = set(archive.getnames())
        required = {'python-research/research.py', 'compiled-research/research.yaml', 'compiled-research/run.sh',
                    'compiled-research/selection.yaml', 'evidence.schema.json', 'catalogue.json'}
        required |= {'inputs/' + name for name in INPUTS}
        required |= {'requirements/research-env.txt', 'requirements/research-macos.txt', 'requirements/build.txt', 'inputs.json', 'recorded_inputs.py', 'commands.json', 'python-research/scenarios.py', 'compiled-research/final/holdout.yaml', 'compiled-research/final/run.sh'}
        required |= {'retained/selection.json', 'retained/python-selection.json', 'retained/compiled-selection.json'}
        required |= {'compiled-research/licenses/TabPFN-code-6.4.1-LICENSE.txt',
                     'compiled-research/licenses/TabPFN-v2-model-LICENSE.txt'}
        if not required <= names:
            raise ValueError('Candidate archive is incomplete')
        if len(names) != len(archive.getmembers()) or any(not member.isfile() for member in archive.getmembers()):
            raise ValueError('Candidate archive has duplicate or non-file members')
        manifest = json.loads(archive_bytes(archive, 'bundle.json'))
        if manifest['schema_version'] != 1 or set(manifest['files']) | {'bundle.json'} != names:
            raise ValueError('Candidate archive inventory differs')
        for name, expected in manifest['files'].items():
            if hashlib.sha256(archive_bytes(archive, name)).hexdigest() != expected:
                raise ValueError('Candidate archive member changed: ' + name)
        archive.extractall(target, filter='data')
    if json.loads((target / 'inputs.json').read_text()) != INPUTS:
        raise ValueError('Candidate input declarations differ')
    canonical = {name: root / 'docs/examples' / name for name in ('recorded_inputs.py', 'commands.json', 'catalogue.json', 'evidence.schema.json')}
    canonical.update({'requirements/research-env.txt': root / 'requirements/ci/research-env.txt',
                      'requirements/research-macos.txt': root / 'docs/examples/requirements-research-macos.txt',
                      'requirements/build.txt': root / 'docs/examples/requirements-build.txt'})
    canonical.update({'retained/' + path.name: path for path in (root / 'docs/examples/retained').glob('*.json')})
    for name, path in canonical.items():
        if (target / name).read_bytes() != path.read_bytes():
            raise ValueError('Canonical bundle member differs: ' + name)
    for name, expected in INPUTS.items():
        if hashlib.sha256((target / 'inputs' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Candidate recorded input differs')
    for interface in ('python', 'compiled'):
        for source in (root / f'docs/examples/{interface}-research').rglob('*'):
            if source.is_file() and source.suffix in ('.py', '.yaml', '.md', '.sh', '.toml', '.txt'):
                relative = source.relative_to(root / 'docs/examples')
                if not (target / relative).is_file() or (target / relative).read_bytes() != source.read_bytes():
                    raise ValueError('Candidate project source differs: ' + str(relative))


def compare_selection_logs(original_directory, command_directory):
    comparisons = []
    for directory in (original_directory, command_directory):
        with (directory / 'results.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2
        comparisons.append({row['id']: float(row['val_backtest_total_return']) for row in rows})
    original, command_line = comparisons
    assert len(original) == len(command_line) == 2 and original.keys() == command_line.keys()
    for key in original:
        assert math.isclose(original[key], command_line[key], rel_tol=1e-10, abs_tol=1e-12), key


def prove_wheel(root: Path, temporary: Path, *, released_wheel: Path | None = None, released_bundle: Path | None = None):
    output = root / '.docs-evidence'
    output.mkdir(exist_ok=True)
    bundle = output / 'examples.tar.gz'
    bundle_sha = build(root, bundle)
    if released_bundle is not None:
        if released_wheel is None or hashlib.sha256(released_bundle.read_bytes()).hexdigest() != bundle_sha:
            raise ValueError('The public project bundle differs from the released research source')
        bundle.write_bytes(released_bundle.read_bytes())
    canonical_commands = json.loads((root / 'docs/examples/commands.json').read_text())
    offline = verify_wheelhouse(root)
    isolated = temporary / 'isolated'
    venv.EnvBuilder(with_pip=True).create(isolated)
    python = isolated / 'bin/python'
    run([python, '-m', 'pip', 'install', '--no-compile', '--no-index', '--find-links', root / '.docs-evidence/wheelhouse',
         '--require-hashes', '-r', root / offline['runtime_lock'], '-r', root / 'docs/examples/requirements-build.txt'], cwd=temporary)
    if released_wheel is None:
        wheel_dir = temporary / 'wheels'
        environment = dict(os.environ)
        environment['SOURCE_DATE_EPOCH'] = subprocess.check_output(['git', 'show', '-s', '--format=%ct', 'HEAD'], cwd=root, text=True).strip()
        environment['PIP_NO_INDEX'] = '1'
        run([python, '-m', 'build', '--wheel', '--no-isolation', '--outdir', wheel_dir], cwd=root, env=environment)
        wheels = list(wheel_dir.glob('*.whl'))
        assert len(wheels) == 1
        wheel = wheels[0]
    else:
        import re
        version = re.search(r'^version = "([^"]+)"$', (root / 'pyproject.toml').read_text(), re.M)
        if version is None:
            raise ValueError('Package version is unavailable')
        wheel = verify_download(released_wheel.resolve(), version[1])
    with zipfile.ZipFile(wheel) as archive:
        shipped = {name for name in archive.namelist() if name.startswith('limen/') and name.endswith('.py')}
        expected = {str(path.relative_to(root)) for path in (root / 'limen').rglob('*.py')}
        assert shipped == expected
        for name in shipped:
            assert archive.read(name) == (root / name).read_bytes()
    runtime_environment = dict(os.environ, PATH=str(python.parent) + os.pathsep + os.environ['PATH'])
    installation_environment = dict(runtime_environment, PIP_NO_INDEX='1',
                                    PIP_FIND_LINKS=wheel.parent.resolve().as_uri() + ' ' + (root / '.docs-evidence/wheelhouse').resolve().as_uri())
    run(shlex.split(canonical_commands['install']), cwd=temporary, env=installation_environment)
    project = temporary / 'bundle'
    project.mkdir()
    unpack(bundle, project, root)
    info_file = temporary / 'installed.json'
    inspect = '''import importlib.metadata as m,json,sys,limen
from pathlib import Path
info={"version":m.version("vaquum-limen"),"import_location":str(Path(limen.__file__).resolve()),"distribution_location":str(m.distribution("vaquum-limen").locate_file("").resolve()),"sys_path":sys.path,"python":sys.version,"dependencies":sorted((d.metadata["Name"],d.version) for d in m.distributions()),"distribution_locations":[str(d.locate_file("").resolve()) for d in m.distributions()]}
Path(sys.argv[1]).write_text(json.dumps(info))'''
    run([python, '-I', '-c', inspect, info_file], cwd=project)
    info = json.loads(info_file.read_text())
    assert info['version'] == wheel.name.split('-')[1]
    if released_wheel is not None:
        assert info['version'] == json.loads(released_wheel.read_text())['version']
    assert Path(info['import_location']).is_relative_to(isolated)
    assert Path(info['distribution_location']).is_relative_to(isolated)
    assert not any(Path(entry or project).resolve() == root.resolve() for entry in info['sys_path'])
    assert all(Path(location).is_relative_to(isolated) for location in info['distribution_locations'])
    from acquisition_proof import prove_acquisition
    acquisition = prove_acquisition(root, temporary, wheel, bundle, runtime_environment, inspect, run)
    summaries, commands, scenario_reports = {}, {}, {}
    assert json.loads((project / 'commands.json').read_text()) == canonical_commands
    # macOS imports compete under host load; Linux's locked CI runs disjoint campaigns together.
    run(shlex.split(canonical_commands['python-selection']), cwd=project, env=runtime_environment)
    with ThreadPoolExecutor(max_workers=1 if sys.platform == 'darwin' else 2) as executor:
        executions = [executor.submit(run, shlex.split(canonical_commands[name]),
                                     cwd=project, env=runtime_environment,
                                     timeout=240 if sys.platform == 'darwin' and name == 'compiled-all' else 120)
                      for name in ('python-all', 'compiled-all')]
        for execution in executions:
            execution.result()
    for interface in ('python',):
        command = shlex.split(canonical_commands[interface + '-selection'])
        summary = json.loads((project / f'results/{interface}/summary.json').read_text())
        assert summary['interface'] == interface and summary['scenario'] == 'selection'
        assert summary['rounds'] == 4
        assert summary['retained_models'] == 4 and summary['inference_rows'] == 480
        destination = output / f'{interface}-selection.json'
        destination.write_text(json.dumps(summary, indent=2) + '\n')
        summaries[interface] = {'summary': summary, 'output_path': str(destination.relative_to(root)), 'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
        commands[interface] = [str(arg) for arg in command]
        scenario_reports[(interface, 'selection')] = summaries[interface] | {'command': commands[interface]}
        command = shlex.split(canonical_commands[interface + '-all'])
        report = json.loads((project / f'results/{interface}-all/summary.json').read_text())
        catalogue = json.loads((project / 'catalogue.json').read_text())
        expected = {name for name, interfaces in catalogue['required_scenarios'].items() if interface in interfaces} - {'selection'}
        assert set(report['scenarios']) == expected
        for name, result in report['scenarios'].items():
            verify(interface, name, result, project / f'results/{interface}-all/{name}', project / 'inputs')
            destination = output / f'{interface}-{name}.json'
            destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            scenario_reports[(interface, name)] = {'summary': result, 'command': [str(arg) for arg in command], 'output_path': str(destination.relative_to(root)), 'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    from cli_proof import verify_project, prove_interrupted_recovery
    command = shlex.split(canonical_commands['compiled-all'])
    compiled = verify_project(project)
    compiled["recovery"].update(prove_interrupted_recovery(project, runtime_environment))
    catalogue = json.loads((project / 'catalogue.json').read_text())
    assert set(compiled) == {name for name, interfaces in catalogue['required_scenarios'].items() if 'compiled' in interfaces}
    for directory, report in [('compiled', compiled['selection']), ('compiled-all', {'scenarios': {key: value for key, value in compiled.items() if key != 'selection'}})]:
        (project / 'results' / directory).mkdir(exist_ok=True)
        (project / 'results' / directory / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, result in compiled.items():
        destination = output / ('compiled-' + name + '.json')
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        scenario_reports[('compiled', name)] = {'summary': result, 'command': command, 'output_path': str(destination.relative_to(root)), 'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    summaries['compiled'] = scenario_reports[('compiled', 'selection')]
    commands['compiled'] = command
    wheel_destination = output / wheel.name
    wheel_destination.write_bytes(wheel.read_bytes())
    native = output / 'native-results.tar.gz'
    native_sha = build_results(project, info_file, native)
    with tarfile.open(native) as retained:
        for name, directory in (('final', 'compiled-holdout'), ('final-event', 'compiled-holdout_event')):
            prefix = f'compiled-research/{name}/manifests/'
            assert prefix + 'holdout-project.json' in retained.getnames()
            assert prefix + 'holdout.sqlite3' in retained.getnames()
            attempt = json.loads((project / 'results/dev' / directory / 'final-report.json').read_text())['attempt_id']
            report = project / prefix / 'holdout' / attempt / 'report.json'
            assert archive_bytes(retained, str(report.relative_to(project))) == report.read_bytes()
    return {'summaries': summaries, 'commands': commands, 'scenario_reports': scenario_reports, 'installed': info, 'offline': offline, 'acquisition': acquisition,
            'native_outputs': {'path': str(native.relative_to(root)), 'sha256': native_sha},
            'bundle': {'path': str(bundle.relative_to(root)), 'sha256': bundle_sha},
            'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
