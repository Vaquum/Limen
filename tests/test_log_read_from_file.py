import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_read_from_file():

    module_path = Path(__file__).resolve().parents[1] / 'limen' / 'log' / '_read_from_file.py'
    spec = importlib.util.spec_from_file_location('_read_from_file_module', module_path)

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module._read_from_file


def test_read_from_file_removes_duplicate_headers_and_trims_object_columns():

    read_from_file = _load_read_from_file()
    csv_text = '\n'.join([
        'recall,label,count',
        '0.10,  first  ,1',
        'recall,label,count',
        'recall,label,count',
        '0.20, second   ,2',
        '0.30,  third,3',
    ]) + '\n'

    with TemporaryDirectory() as tmpdir:

        input_path = Path(tmpdir) / 'experiment.csv'
        input_path.write_text(csv_text, encoding='utf-8')

        data = read_from_file(None, str(input_path))

    assert list(data.columns) == ['recall', 'label', 'count']
    assert data['recall'].tolist() == [0.1, 0.2, 0.3]
    assert data['label'].tolist() == ['first', 'second', 'third']
    assert data['count'].tolist() == [1, 2, 3]
