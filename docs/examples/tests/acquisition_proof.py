from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
from threading import Thread


def prove_acquisition(root, temporary, wheel, bundle, environment, inspect_program, run_command):
    served = temporary / 'published-artifacts'
    served.mkdir()
    files = served / 'evidence'
    files.mkdir()
    identities = {}
    for name, source in (('examples', bundle), ('wheel', wheel)):
        shutil.copyfile(source, files / source.name)
        identities[name] = {'path': 'evidence/' + source.name,
                            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest = {'release': 'v' + wheel.name.split('-')[1], 'artifacts': identities}
    (served / 'publication.json').write_text(json.dumps(manifest))
    consumer = temporary / 'consumer'
    consumer.mkdir()
    shutil.copyfile(root / 'docs/examples/acquire.sh', consumer / 'acquire.sh')
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(served)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = dict(environment, LIMEN_DOCS_ORIGIN='http://127.0.0.1:' + str(server.server_port),
                   PIP_NO_INDEX='1', PIP_FIND_LINKS=str((root / '.docs-evidence/wheelhouse').resolve()))
        commands = json.loads((root / 'docs/examples/commands.json').read_text())
        assert commands['acquire-project'] == 'sh acquire.sh limen-research'
        run_command(['sh', 'acquire.sh', 'limen-research'], cwd=consumer, env=env, timeout=90)
        acquired = consumer / 'limen-research'
        assert json.loads((acquired / 'publication.json').read_text()) == manifest
        assert (acquired / 'python-research/research.py').read_bytes() == (root / 'docs/examples/python-research/research.py').read_bytes()
        assert (acquired / 'compiled-research/research.yaml').read_bytes() == (root / 'docs/examples/compiled-research/research.yaml').read_bytes()
        for name, digest in json.loads((acquired / 'inputs.json').read_text()).items():
            assert hashlib.sha256((acquired / 'inputs' / name).read_bytes()).hexdigest() == digest
        receipt = acquired / 'installed.json'
        run_command([acquired / '.venv/bin/python', '-I', '-c', inspect_program, receipt], cwd=acquired, env=env)
        installed = json.loads(receipt.read_text())
        assert installed['version'] == wheel.name.split('-')[1]
        assert Path(installed['import_location']).is_relative_to(acquired / '.venv')
        assert all(Path(location).is_relative_to(acquired / '.venv') for location in installed['distribution_locations'])
        # A changed real archive must fail before extraction or installation.
        (files / bundle.name).write_bytes(bundle.read_bytes() + b'corrupted')
        failed = subprocess.run(['sh', 'acquire.sh', 'changed-artifact'], cwd=consumer, env=env, check=False)
        assert failed.returncode != 0
        assert not (consumer / 'changed-artifact/.venv').exists()
        return {'release': manifest['release'], 'wheel_sha256': identities['wheel']['sha256'],
                'bundle_sha256': identities['examples']['sha256'], 'changed_artifact_rejected': True,
                'installed_outside_checkout': True}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
