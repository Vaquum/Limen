import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import venv
import zipfile

from build_bundle import INPUTS, build, build_results
from prepare_environment import verify_wheelhouse
from scenario_oracles import verify


def run(argv, *, cwd, env=None):
    environment = dict(os.environ if env is None else env)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        environment.pop(name, None)
    environment['PYTHONNOUSERSITE'] = '1'
    subprocess.run([str(arg) for arg in argv], cwd=cwd, env=environment, check=True, timeout=90)


def archive_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError('Missing regular archive file: ' + name)
    with stream:
        return stream.read()


def unpack(bundle: Path, target: Path, root: Path):
    with tarfile.open(bundle) as archive:
        names = set(archive.getnames())
        required = {'python-research/research.py', 'compiled-research/research.py',
                    'compiled-research/selection.yaml', 'evidence.schema.json', 'catalogue.json'}
        required |= {'inputs/' + name for name in INPUTS}
        required |= {'requirements/research-env.txt', 'inputs.json', 'recorded_inputs.py', 'python-research/scenarios.py', 'compiled-research/scenarios.py'}
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
    if (target / 'recorded_inputs.py').read_bytes() != (root / 'docs/examples/recorded_inputs.py').read_bytes():
        raise ValueError('Shared recorded-input source differs')
    for name, expected in INPUTS.items():
        if hashlib.sha256((target / 'inputs' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Candidate recorded input differs')
    for interface in ('python', 'compiled'):
        for source in (root / f'docs/examples/{interface}-research').rglob('*'):
            if source.is_file() and source.suffix in ('.py', '.yaml', '.md'):
                relative = source.relative_to(root / 'docs/examples')
                if not (target / relative).is_file() or (target / relative).read_bytes() != source.read_bytes():
                    raise ValueError('Candidate project source differs: ' + str(relative))


def prove_wheel(root: Path, temporary: Path):
    output = root / '.docs-evidence'
    output.mkdir(exist_ok=True)
    bundle = output / 'examples.tar.gz'
    bundle_sha = build(root, bundle)
    offline = verify_wheelhouse(root)
    isolated = temporary / 'isolated'
    venv.EnvBuilder(with_pip=True).create(isolated)
    python = isolated / 'bin/python'
    run([python, '-m', 'pip', 'install', '--no-index', '--find-links', root / '.docs-evidence/wheelhouse',
         '--require-hashes', '-r', root / offline['runtime_lock'], '-r', root / 'docs/examples/requirements-build.txt'], cwd=temporary)
    wheel_dir = temporary / 'wheels'
    environment = dict(os.environ)
    environment['SOURCE_DATE_EPOCH'] = subprocess.check_output(['git', 'show', '-s', '--format=%ct', 'HEAD'], cwd=root, text=True).strip()
    environment['PIP_NO_INDEX'] = '1'
    run([python, '-m', 'build', '--wheel', '--no-isolation', '--outdir', wheel_dir], cwd=root, env=environment)
    wheels = list(wheel_dir.glob('*.whl'))
    assert len(wheels) == 1
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        shipped = {name for name in archive.namelist() if name.startswith('limen/') and name.endswith('.py')}
        expected = {str(path.relative_to(root)) for path in (root / 'limen').rglob('*.py')}
        assert shipped == expected
        for name in shipped:
            assert archive.read(name) == (root / name).read_bytes()
    run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', wheel], cwd=temporary)
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
    assert Path(info['import_location']).is_relative_to(isolated)
    assert Path(info['distribution_location']).is_relative_to(isolated)
    assert not any(Path(entry or project).resolve() == root.resolve() for entry in info['sys_path'])
    assert all(Path(location).is_relative_to(isolated) for location in info['distribution_locations'])
    summaries, commands, scenario_reports = {}, {}, {}
    for interface in ('python', 'compiled'):
        command = [python, f'{interface}-research/research.py', '--scenario', 'selection', '--data-dir', 'inputs', '--output-dir', f'results/{interface}']
        run(command, cwd=project)
        summary = json.loads((project / f'results/{interface}/summary.json').read_text())
        assert summary['interface'] == interface and summary['scenario'] == 'selection'
        assert summary['rounds'] == (4 if interface == 'python' else 2)
        if interface == 'python':
            assert summary['retained_models'] == 4 and summary['inference_rows'] == 480
        else:
            assert summary['reconstructed_sensors'] == summary['selected_rounds'] == 1
        destination = output / f'{interface}-selection.json'
        destination.write_text(json.dumps(summary, indent=2) + '\n')
        summaries[interface] = {'summary': summary, 'output_path': str(destination.relative_to(root)), 'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
        commands[interface] = [str(arg) for arg in command]
        scenario_reports[(interface, 'selection')] = summaries[interface] | {'command': commands[interface]}
        command = [python, f'{interface}-research/research.py', '--scenario', 'all', '--data-dir', 'inputs', '--output-dir', f'results/{interface}-all']
        run(command, cwd=project)
        report = json.loads((project / f'results/{interface}-all/summary.json').read_text())
        catalogue = json.loads((project / 'catalogue.json').read_text())
        expected = {name for name, interfaces in catalogue['required_scenarios'].items() if interface in interfaces} - {'selection'}
        assert set(report['scenarios']) == expected
        for name, result in report['scenarios'].items():
            verify(interface, name, result, project / f'results/{interface}-all/{name}', project / 'inputs')
            destination = output / f'{interface}-{name}.json'
            destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            scenario_reports[(interface, name)] = {'summary': result, 'command': [str(arg) for arg in command], 'output_path': str(destination.relative_to(root)), 'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    wheel_destination = output / wheel.name
    wheel_destination.write_bytes(wheel.read_bytes())
    native = output / 'native-results.tar.gz'
    native_sha = build_results(project, info_file, native)
    return {'summaries': summaries, 'commands': commands, 'scenario_reports': scenario_reports, 'installed': info, 'offline': offline,
            'native_outputs': {'path': str(native.relative_to(root)), 'sha256': native_sha},
            'bundle': {'path': str(bundle.relative_to(root)), 'sha256': bundle_sha},
            'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
