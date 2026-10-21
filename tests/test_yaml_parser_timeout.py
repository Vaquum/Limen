import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from limen.yaml.parser import parse


# Exact ConsumeUnicodeNoSurrogates(4096) outputs from the retained units in #1131.
@pytest.mark.parametrize('text,text_sha256,column', [
    pytest.param(
        '{{{{{{{\r\n{{{{{{{{{{{\r\n{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}',
        '2262c73026f9ab49a9f35cd190f8ef535d72ba4ba9aefd2133271b1014d67eb6',
        2,
        id='timeout-ee68e899926153ddb692d6be630bc31405e25d3a',
    ),
    pytest.param(
        '?    ?              ?             ?    ?     ?               ?    ?               ?          ?    ?     ?               ?    ?               ?     ?     ',
        'f9002c7db21cd4257729e573263af2619f04f69243ffb960a0f2a3d37aed43b9',
        6,
        id='timeout-8f5d372403f4914811126ed38347bc889bf25157',
    ),
])
def test_retained_mapping_key_timeouts(text: str, text_sha256: str, column: int, tmp_path: Path) -> None:
    assert hashlib.sha256(text.encode()).hexdigest() == text_sha256
    completed = subprocess.run(
        [sys.executable, '-c', (
            'import json, sys; from dataclasses import asdict; '
            'from limen.yaml.parser import parse; '
            'data, errors = parse(json.load(sys.stdin)); '
            'print(json.dumps({"data": data, "errors": [asdict(error) for error in errors]}))'
        )],
        input=json.dumps(text),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    result = json.loads(completed.stdout)
    path = tmp_path / 'retained.yaml'
    path.write_text(text, encoding='utf-8')
    for source in (text, path):
        parsed, errors = parse(source)
        assert parsed == result['data'] == {}
        assert [asdict(error) for error in errors] == result['errors']
        assert len(errors) == 1
        assert 'mappings cannot be YAML keys' in errors[0].message
        assert (errors[0].line, errors[0].column) == (1, column)
