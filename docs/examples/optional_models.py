"""Capture and verify the separately provisioned real-model CLI observation."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import polars as pl

MODEL_SHA256 = 'cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
SOURCES = ('compiled-research/tabpfn.yaml', 'compiled-research/tabpfn.sh')
INPUT = 'spot_15m_20250101_20250531.parquet'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_sources(directory):
    return {str(path.relative_to(directory)): digest(path) for path in sorted(directory.rglob('*.py'))}


def verify(root, record):
    assert record['package_sources'] == package_sources(root / 'limen')
    assert record['sources'] == {name: digest(root / 'docs/examples' / name) for name in SOURCES}
    version = re.search(r'^version = "([^"]+)"$', (root / 'pyproject.toml').read_text(), re.M)
    assert version is not None and record['package_version'] == version[1]
    assert record['input_sha256'] == digest(root / 'tests/fixtures' / INPUT)
    assert record['model_sha256'] == MODEL_SHA256 and record['dependencies']['tabpfn'] == '6.4.1'
    raw = pl.read_parquet(root / 'tests/fixtures' / INPUT)
    train = raw.filter(pl.col('datetime') < datetime(2025, 1, 3, tzinfo=timezone.utc))['close'].to_numpy()
    cutoff = np.median((train[1:] / train[:-1] - 1) * 100)
    labels = raw.with_columns((((pl.col('close').shift(-1) / pl.col('close') - 1) * 100) > cutoff).cast(pl.Int8).alias('label'))
    aligned = labels.filter(pl.col('datetime').is_between(datetime.fromisoformat(record['alignment']['first_test_datetime']), datetime.fromisoformat(record['alignment']['last_test_datetime'])))
    truth, predicted, probabilities = aligned['label'].to_numpy(), np.asarray(record['predictions']), np.asarray(record['probabilities'])
    assert len(truth) == len(predicted) == len(probabilities) == 95
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(predicted, probabilities >= .5)
    assert record['metrics']['accuracy'] == float(np.mean(truth == predicted))
    differences = probabilities[truth == 1, None] - probabilities[truth == 0]
    auc = float(np.mean((differences > 0) + .5 * (differences == 0)))
    np.testing.assert_allclose(record['metrics']['auc'], auc, atol=1e-12)


def capture(root, project, output):
    import limen
    from limen.experiment._round_storage import open_text
    assert {name: digest(project / name) for name in SOURCES} == {name: digest(root / 'docs/examples' / name) for name in SOURCES}
    installed = Path(limen.__file__).resolve().parent
    assert not installed.is_relative_to(root.resolve())
    assert package_sources(installed) == package_sources(root / 'limen')
    environment = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        environment.pop(name, None)
    environment['PYTHONNOUSERSITE'] = '1'
    subprocess.run(['sh', 'compiled-research/tabpfn.sh'], cwd=project, env=environment, check=True, timeout=180)
    location = project / 'results/dev/compiled-family-tabpfn'
    log = pl.read_csv(location / 'results.csv')
    assert log.height == 1 and log['strict_mode_error'].drop_nulls().len() == 0
    with open_text(location / 'round_data.jsonl.gz') as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 1
    row = rows[0]
    record = {'observed_at': datetime.now(timezone.utc).isoformat(),
        'execution_checkout_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'package_version': metadata.version('vaquum-limen'), 'package_sources': package_sources(installed),
        'sources': {name: digest(project / name) for name in SOURCES},
        'input_sha256': digest(project / 'inputs' / INPUT),
        'model_sha256': digest(project / 'tabpfn-v2-classifier-v2_default.ckpt'),
        'dependencies': {name: metadata.version(name) for name in ('tabpfn', 'torch', 'numpy', 'scikit-learn', 'polars', 'pyarrow')},
        'alignment': row['alignment'], 'predictions': row['preds'], 'probabilities': row['probs'],
        'metrics': {name: log[name][0] for name in ('accuracy', 'auc')},
        'scope': 'Separate installed candidate wheel, real pinned checkpoint and unchanged YAML/CLI; not a fresh optional-model run in ordinary CI.'}
    verify(root, record)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print('Verified real TabPFN CLI predictions, probabilities and independent recorded-label metrics: ' + str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture(args.root, args.project, args.output)
