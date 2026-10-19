import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


INPUTS = {
    'spot_1h_20240101_20241231.parquet': 'a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72',
    'spot_15m_20250101_20250531.parquet': 'eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a',
    'btcusdt_funding_20240101_20240108.json': '4352c58c5467ba48b5d13039052085e3bd59c089d6e1455bf99f4063279c3df2',
    'binance_trades_btcusdt_2025-05-23_90s.zip': 'ecc0d2c54aad5c69bba510fe0453d91b288f23579e568d665c06ecfae49757b5',
    'observed_costs/benchmark.json': 'b71011697713417b21302e670abf78c0bff739ec9135c7012db02fa3b2d6ccf8',
    'observed_costs/history.zip': 'df4d7776172df459ec297cbcfe8e72a4748a51ab626e3c990deec83415102093',
    'observed_costs/license.json': '23484080def908bce03a3ad470c783ad4e7aea821b4275f16a132eec737f69d7',
    'observed_costs/provenance.json': 'e533eee80a6e90fd8d30936795a3fbbcb25b9d7080d57e8f11880f5ed3765e98',
    'observed_costs/reproduce.py': '3fa7c01d291c119b03fa1040673904b06ffb8d3e00f81eb3daa55b13a1f2adf0',
}


def write_archive(entries: dict[str, bytes], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as compressed, tarfile.open(fileobj=compressed, mode='w|') as archive:
        for name, content in sorted(entries.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(content), 0o644
            archive.addfile(member, io.BytesIO(content))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def build_results(project: Path, environment: Path, output: Path) -> str:
    entries = {'environment/installed.json': environment.read_bytes()}
    for path in sorted((project / 'results').rglob('*')):
        if path.is_symlink():
            raise ValueError('Native results cannot substitute an external file: ' + str(path))
        if path.is_file():
            entries[str(path.relative_to(project))] = path.read_bytes()
    entries['files.json'] = (json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}, indent=2) + '\n').encode()
    return write_archive(entries, output)


def build(root: Path, output: Path) -> str:
    entries = {}
    for interface in ('python', 'compiled'):
        for path in (root / f'docs/examples/{interface}-research').rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.md'):
                entries[f'{interface}-research/{path.relative_to(root / f"docs/examples/{interface}-research")}'] = path.read_bytes()
    for name in ('catalogue.json', 'evidence.schema.json', 'recorded_inputs.py', 'commands.json'):
        entries[name] = (root / 'docs/examples' / name).read_bytes()
    for name, expected in INPUTS.items():
        data = (root / 'tests/fixtures' / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Recorded input digest differs: {name}')
        entries['inputs/' + name] = data
    entries['requirements/research-env.txt'] = (root / 'requirements/ci/research-env.txt').read_bytes()
    entries['requirements/research-macos.txt'] = (root / 'docs/examples/requirements-research-macos.txt').read_bytes()
    entries['requirements/build.txt'] = (root / 'docs/examples/requirements-build.txt').read_bytes()
    entries['inputs.json'] = (json.dumps(INPUTS, indent=2) + '\n').encode()
    entries['bundle.json'] = (json.dumps({'schema_version': 1, 'files': {name: hashlib.sha256(content).hexdigest() for name, content in sorted(entries.items())}}, indent=2) + '\n').encode()
    return write_archive(entries, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the pinned research project/input bundle.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[2], args.output))
