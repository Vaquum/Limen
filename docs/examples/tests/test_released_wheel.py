import copy
import json
from pathlib import Path

import pytest

from released_wheel import sha256, verify_download, wheel_entry

METADATA = json.loads((Path(__file__).parent / 'fixtures/pypi-v5.45.3.json').read_text())


def test_recorded_public_release_selects_its_actual_wheel():
    entry = wheel_entry(METADATA, '5.45.3')
    assert entry['filename'] == 'vaquum_limen-5.45.3-py3-none-any.whl'
    assert entry['digests']['sha256'] == 'ce4576c25493bb9bed77d6ff7e0b718da31111c465576badac91a33362d103fe'
    assert entry['size'] == 1119368


@pytest.mark.parametrize('change', ['package', 'version', 'yanked', 'duplicate', 'host', 'digest'])
def test_unavailable_or_substituted_release_cannot_be_selected(change):
    metadata = copy.deepcopy(METADATA)
    entry = metadata['urls'][0]
    if change == 'package':
        metadata['info']['name'] = 'another-package'
    elif change == 'version':
        metadata['info']['version'] = '5.45.2'
    elif change == 'yanked':
        entry['yanked'] = True
    elif change == 'duplicate':
        metadata['urls'].append(copy.deepcopy(entry))
    elif change == 'host':
        entry['url'] = entry['url'].replace('files.pythonhosted.org', 'example.invalid')
    else:
        entry['digests']['sha256'] = 'unavailable'
    with pytest.raises(ValueError):
        wheel_entry(metadata, '5.45.3')


def test_download_receipt_cannot_substitute_version_metadata_or_wheel(tmp_path):
    metadata = json.dumps(METADATA).encode()
    (tmp_path / 'pypi-5.45.3.json').write_bytes(metadata)
    entry = METADATA['urls'][0]
    record = {'release': 'v5.45.3', 'version': '5.45.3', 'filename': entry['filename'],
              'wheel_sha256': entry['digests']['sha256'], 'metadata_sha256': sha256(metadata), 'url': entry['url']}
    receipt = tmp_path / 'released-wheel.json'
    receipt.write_text(json.dumps(record))
    (tmp_path / entry['filename']).write_bytes(metadata)
    with pytest.raises(ValueError, match='changed after download'):
        verify_download(receipt, '5.45.3')
    for field in ('version', 'metadata_sha256', 'wheel_sha256', 'filename', 'url'):
        receipt.write_text(json.dumps(record | {field: 'substituted'}))
        with pytest.raises(ValueError, match='receipt'):
            verify_download(receipt, '5.45.3')
