import os
from pathlib import Path
from tempfile import TemporaryDirectory

from limen import Log


def _read_log(csv_text: str, *, change_cwd: bool = False):

    with TemporaryDirectory() as tmpdir:

        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / 'experiment.csv'
        input_path.write_text(csv_text, encoding='utf-8')
        temp_path = tmpdir_path / '__temp__.csv'
        original_cwd = Path.cwd()

        try:
            if change_cwd:
                os.chdir(tmpdir_path)

            log = Log(file_path=str(input_path))
            data = log.experiment_log.copy()

        finally:
            if change_cwd:
                os.chdir(original_cwd)

    return data, temp_path


def test_read_from_file_removes_duplicate_headers_and_trims_object_columns():

    csv_text = '\n'.join([
        'recall,label,count',
        '0.10,  first  ,1',
        'recall,label,count',
        'recall,label,count',
        '0.20, second   ,2',
        '0.30,  third,3',
    ]) + '\n'

    data, _ = _read_log(csv_text)

    assert list(data.columns) == ['recall', 'label', 'count']
    assert data['recall'].tolist() == [0.1, 0.2, 0.3]
    assert data['label'].tolist() == ['first', 'second', 'third']
    assert data['count'].tolist() == [1, 2, 3]


def test_read_from_file_matches_full_header_and_keeps_non_header_recall_rows():

    csv_text = '\n'.join([
        'id,label,count',
        'alpha-1,  first  ,1',
        'id,label,count',
        'recall-row, second   ,2',
        'id,label,count   ',
        'beta-2,  third,3',
    ]) + '\n'

    data, temp_path = _read_log(csv_text, change_cwd=True)

    assert list(data.columns) == ['id', 'label', 'count']
    assert data['id'].tolist() == ['alpha-1', 'recall-row', 'beta-2']
    assert data['label'].tolist() == ['first', 'second', 'third']
    assert data['count'].tolist() == [1, 2, 3]
    assert not temp_path.exists()
