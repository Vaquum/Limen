from importlib.util import find_spec
from pathlib import Path

TEMPLATES_DIR = Path(find_spec('limen.yaml').origin).parent / 'templates'
