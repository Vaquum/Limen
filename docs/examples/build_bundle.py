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
}


def build(root: Path, output: Path) -> str:
    entries = {}
    for interface in ('python', 'compiled'):
        for path in (root / f'docs/examples/{interface}-research').rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.md'):
                entries[f'{interface}-research/{path.relative_to(root / f"docs/examples/{interface}-research")}'] = path.read_bytes()
    for name in ('catalogue.json', 'evidence.schema.json', 'check_evidence.py'):
        entries[name] = (root / 'docs/examples' / name).read_bytes()
    for name, expected in INPUTS.items():
        data = (root / 'tests/fixtures' / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Recorded input digest differs: {name}')
        entries['inputs/' + name] = data
    entries['requirements/research-env.txt'] = (root / 'requirements/ci/research-env.txt').read_bytes()
    entries['inputs.json'] = (json.dumps(INPUTS, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as compressed, tarfile.open(fileobj=compressed, mode='w|') as archive:
        for name, content in sorted(entries.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(content), 0o644
            archive.addfile(member, io.BytesIO(content))
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the pinned research project/input bundle.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[2], args.output))
