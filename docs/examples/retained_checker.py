import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import sys
from uuid import uuid4

import provenance


def checker_identity():
    expected = Path(__file__).resolve().with_name('provenance.py')
    if Path(provenance.__file__).resolve() != expected or sys.modules.get('provenance') is not provenance:
        raise ValueError('The retained checker did not load its reviewed provenance dependency')
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), expected)}


def product_identity(root):
    if Path(provenance.git(root, 'rev-parse', '--show-toplevel')).resolve() != root:
        raise ValueError('Run the retained checker from the product checkout root')
    if provenance.git(root, 'status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('The retained product checkout must remain clean')
    return {'head': provenance.git(root, 'rev-parse', 'HEAD'),
            'tree': provenance.git(root, 'rev-parse', 'HEAD^{tree}')}


def install_launcher(destination):
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Retain existing launchers; install to a new path')
    checker_identity()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = '#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(Path(__file__).resolve())) + ' "$@"\n'
    with destination.open('x') as stream:
        stream.write(payload)
    destination.chmod(0o700)
    print(json.dumps({'launcher': str(destination), 'python': sys.executable,
                      'launcher_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                      'checker': checker_identity()}, indent=2))


def execute(root, arguments):
    sys.path[:0] = [str(root / 'docs/examples/tests'), str(root / 'docs/examples')]
    if arguments[:2] == ['-m', 'pytest']:
        import pytest
        return int(pytest.main(arguments[2:]))
    if not arguments:
        raise ValueError('Name the original provenance, pytest or evidence check')
    script = Path(arguments[0]).resolve()
    if script == root / 'docs/examples/provenance.py':
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', type=Path, required=True)
        selected = parser.parse_args(arguments[1:])
        record = provenance.acquire(root, selected.output)
        provenance.validate_provenance(root, record)
        print(f"Bound {len(record['files'])} retained sources with the reviewed checker.")
        return 0
    if script != root / 'docs/examples/check_evidence.py':
        raise ValueError('The retained checker only runs the original documentation checks')
    sys.argv = [str(script), *arguments[1:]]
    runpy.run_path(str(script), run_name='__main__')
    return 0


def main():
    if sys.argv[1:2] == ['--install-launcher']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--install-launcher', type=Path, required=True)
        install_launcher(parser.parse_args().install_launcher)
        return 0
    root = Path.cwd().resolve()
    product, checker = product_identity(root), checker_identity()
    output = root / '.docs-evidence/retained-checker'
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / (str(uuid4()) + '.json')
    record = {'kind': 'checker-invocation', 'product': product, 'checker': checker,
              'python': sys.executable, 'arguments': sys.argv[1:], 'exit_code': None}
    print('Retained documentation checker: ' + json.dumps(record), flush=True)
    try:
        code = execute(root, sys.argv[1:])
        if product_identity(root) != product or checker_identity() != checker:
            raise ValueError('Product or checker identity changed during validation')
        record['exit_code'] = code
        return code
    except SystemExit as error:
        record['exit_code'] = error.code if isinstance(error.code, int) else int(error.code is not None)
        raise
    except KeyboardInterrupt:
        record['exit_code'] = 130
        raise
    except BaseException:
        record['exit_code'] = 1
        raise
    finally:
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print('Checker invocation receipt: ' + str(receipt), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
