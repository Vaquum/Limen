import json
from pathlib import Path


def test_taxonomy_covers_public_definitions_once_with_valid_categories() -> None:
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / 'datasets' / 'taxonomy.json').read_text())
    categories = {
        'trend', 'momentum', 'volatility', 'volume',
        'market_structure', 'trade_logic', 'temporal', 'exotic',
    }
    expected = {
        (kind, path.stem)
        for kind in ('indicator', 'feature')
        for path in (root / 'limen' / f'{kind}s').glob('*.py')
        if not path.stem.startswith('_')
    }

    assert isinstance(entries, list)
    for entry in entries:
        assert set(entry) == {'name', 'type', 'category'}, entry
        assert entry['category'] in categories, entry
    keys = [(entry['type'], entry['name']) for entry in entries]
    assert len(keys) == len(set(keys)), 'Duplicate taxonomy definitions'
    assert set(keys) == expected, {
        'missing': sorted(expected - set(keys)),
        'unknown': sorted(set(keys) - expected),
    }
