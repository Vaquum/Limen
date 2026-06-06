import tempfile
from pathlib import Path

from limen.yaml.config import find_project_root
from limen.yaml.config import get_store_path
from limen.yaml.config import read_limen_toml
from limen.yaml.config import _STORE_RELATIVE


def test_find_project_root_walks_up_from_subdirectory() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / 'limen.toml').write_text('')
        subdir = root / 'manifests' / 'examples'
        subdir.mkdir(parents=True)
        assert find_project_root(subdir) == root


def test_find_project_root_returns_none_when_not_found() -> None:
    with tempfile.TemporaryDirectory() as d:
        assert find_project_root(Path(d)) is None


def test_get_store_path_raises_when_no_project_root() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            get_store_path(Path(d))
            assert False, 'expected FileNotFoundError'
        except FileNotFoundError:
            pass


def test_get_store_path_returns_committed_dir() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / 'limen.toml').write_text('')
        assert get_store_path(root) == root / _STORE_RELATIVE


def test_read_limen_toml_returns_parsed_contents() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / 'limen.toml').write_text('[project]\nname = "my-project"\n')
        data = read_limen_toml(root)
        assert data['project']['name'] == 'my-project'


def test_read_limen_toml_raises_when_file_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            read_limen_toml(Path(d))
            assert False, 'expected FileNotFoundError'
        except FileNotFoundError:
            pass
