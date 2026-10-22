import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from threading import Thread
import venv

from publication import write_artifact
from verify_publication import fetch, sha256

ROOT = Path(__file__).resolve().parents[3]


def assert_installable_download(wheel: Path, temporary: Path):
    output, site = temporary / 'release', temporary / 'site'
    output.mkdir(parents=True)
    site.mkdir()
    original = wheel.read_bytes()
    artifact = write_artifact(output, site, 'wheel', original, 'whl', wheel_name=wheel.name)
    assert artifact == {'path': 'evidence/wheel-' + sha256(original) + '/' + wheel.name,
                        'sha256': sha256(original)}
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(site))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        downloaded = temporary / wheel.name
        downloaded.write_bytes(fetch(f'http://127.0.0.1:{server.server_port}/' + artifact['path']))
        assert downloaded.read_bytes() == (output / wheel.name).read_bytes() == original
        isolated = temporary / 'installed'
        venv.EnvBuilder(with_pip=True).create(isolated)
        python = isolated / 'bin/python'
        subprocess.run([str(python), '-I', '-m', 'pip', 'install', '--no-index', '--no-deps', str(downloaded)],
                       cwd=temporary, check=True, timeout=30)
        inspect = subprocess.check_output([str(python), '-I', '-c',
            'from importlib.metadata import distribution; import json; d=distribution("vaquum-limen"); print(json.dumps({"version":d.version,"location":str(d.locate_file(""))}))'],
            cwd=temporary, text=True)
        metadata = json.loads(inspect)
        assert Path(metadata['location']).resolve().is_relative_to(isolated.resolve())
        assert downloaded.name.startswith('vaquum_limen-' + metadata['version'] + '-')
        return metadata
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_published_wheel_keeps_installable_distribution_filename(research_proof, tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    wheels = list((ROOT / '.docs-evidence').glob('vaquum_limen-' + version('vaquum-limen') + '-*.whl'))
    assert len(wheels) == 1
    assert assert_installable_download(wheels[0], tmp_path)['version'] == version('vaquum-limen')
