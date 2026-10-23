import json
from pathlib import Path

import pytest

from optional_models import verify

ROOT = Path(__file__).resolve().parents[3]


def test_retained_tabpfn_matches_current_contract():
    record = json.loads((ROOT / 'docs/examples/retained/tabpfn.json').read_text())
    verify(ROOT, record)
    record['sources']['compiled-research/tabpfn.yaml'] = '0' * 64
    with pytest.raises(AssertionError):
        verify(ROOT, record)
