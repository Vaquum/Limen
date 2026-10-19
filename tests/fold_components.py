import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time

import polars as pl
import pytest
from threadpoolctl import threadpool_limits
import limen.data as _data_namespace
import limen.sfd.reference_architecture as _architecture_namespace

from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.inference import Trainer
from limen.sfd.reference_architecture import LogRegBinary
from limen.yaml import CompiledSFD, build_search_strategy, parse

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'
FULL = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)


class CallerBinary(LogRegBinary):
    inference_output = ('binary', True)


def caller_binary(data: dict[str, object], **params: object) -> dict[str, object]:
    model = CallerBinary().train(data, **params)
    return {**model.evaluate(data), '_model': model}


def sorted_source(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort('datetime').unique('datetime', keep='first', maintain_order=True)


# Install real caller components in the namespaces admitted by Compiled Research.
vars(_architecture_namespace)['_fold_test_binary'] = caller_binary
vars(_data_namespace)['_fold_test_sorted_source'] = sorted_source


@dataclass
class Campaign:
    path: Path
    bars: pl.DataFrame
    config: dict
    metadata: dict
    record: dict
    sensor: object
    fold: int
    elapsed: float


def config(kind: str, source: Path) -> dict:
    reference, errors = parse((ROOT / 'limen/yaml/templates/ridge_regressor.yaml').read_text())
    assert not errors
    manifest = reference['sfd']['manifest']
    manifest.pop('split_dates')
    manifest['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(source)}}
    manifest['split_walk_forward'] = {'n_folds': 5 if kind in ('anchored', 'rolling') else 2,
                                    'test_bars': 48, 'purge_bars': 4, 'embargo_bars': 3,
                                    'anchored': kind != 'rolling'}
    if kind != 'continuous':
        manifest['target'] = {'name': 'quantile_flag', 'class': 'limen.targets.QuantileBinaryTarget',
                              'fit_params': {'source_column': 'ret_1', 'quantile': 0.5}, 'transform_params': {'shift': -1}}
    architecture = 'ridge_regressor' if kind == 'continuous' else 'lightgbm_binary' if kind == 'rolling' else 'logreg_binary'
    manifest['reference_architecture'] = 'limen.sfd.reference_architecture.' + architecture
    parameters = {'alpha': [1.0], 'solver': ['svd']} if kind == 'continuous' else {'C': [1.0], 'max_iter': [1000], 'random_state': [42], 'n_jobs': [1]}
    if kind == 'rolling':
        parameters = {'learning_rate': [0.1], 'n_estimators': [10], 'num_leaves': [5], 'verbosity': [-1], 'n_jobs': [1], 'random_state': [42]}
    if kind in ('anchored', 'rolling'):
        manifest['calibration'] = {'probability_calibration': {'func': 'limen.calibration.sklearn_probability_calibrator', 'params': {'method': 'sigmoid'}},
                                   'threshold_function': {'func': 'limen.calibration.grid_threshold_optimizer', 'params': {'metric': 'limen.metrics.balanced_metric.balanced_metric', 'threshold_min': 0.5, 'threshold_max': 0.5, 'threshold_step': 0.1}}}
    if kind == 'custom':
        manifest['reference_architecture'] = 'limen.sfd.reference_architecture._fold_test_binary'
    if kind in ('reordered', 'duplicates'):
        manifest['pre_split_data_selector'] = {'func': 'limen.data._fold_test_sorted_source'}
    if kind == 'formed':
        manifest['bar_formation'] = {'func': 'limen.data.bars.trade_bars', 'params': {'trade_threshold': 100000}}
        parameters['bar_type'] = ['trade']
    if kind == 'signed':
        from tests.test_direction_sizing import YAML_SPEC
        reference, errors = parse(YAML_SPEC)
        assert not errors
        manifest = reference['sfd']['manifest']
        manifest.pop('split_dates')
        manifest['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(source)}}
        manifest['split_walk_forward'] = {'n_folds': 2, 'test_bars': 48, 'purge_bars': 4, 'embargo_bars': 3, 'anchored': True}
        parameters = reference['sfd']['params']
        parameters.update(holding=[3600], folds=[2])
    reference['sfd']['params'] = parameters
    reference['uel'].update(n_permutations=1, search_strategy={'type': 'grid'}, prep_each_round=True)
    return reference


@pytest.fixture(scope='session')
def fold_campaigns(tmp_path_factory):
    result = {}
    root = tmp_path_factory.mktemp('fold-campaigns')
    original = pl.read_parquet(FIXTURE)
    for kind in ('anchored', 'rolling', 'continuous', 'custom', 'formed', 'parallel', 'signed', 'reordered', 'duplicates', 'companion', 'ordinary'):
        started = time.perf_counter()
        source = root / 'source.parquet'
        count = 432 if kind in ('anchored', 'rolling') else 384
        bars = original.slice(432 if kind == 'companion' else 0, count)
        if kind == 'signed':
            bars = bars.with_columns(pl.lit(3600).alias('base_interval'))
        if kind == 'reordered':
            bars = pl.concat([bars[:300], bars[301:384], bars[300:301]])
        if kind == 'duplicates':
            bars = pl.concat([bars, bars[300:301]])
        bars.write_parquet(source)
        reference = config('custom' if kind == 'companion' else kind, source)
        if kind == 'ordinary':
            reference['sfd']['manifest'].pop('split_walk_forward')
        if kind == 'parallel':
            reference['sfd']['params']['C'] = [0.1, 1.0]
            reference['uel']['n_permutations'] = 2
        output = root / 'output'
        loop = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=bars, experiment_dir=output,
                                       yaml_reference=reference, search_strategy=build_search_strategy(reference), checkpoint_interval=1)
        if kind in ('anchored', 'rolling'):
            loop.manifest.set_split_config(*(4, 1, 2) if kind == 'anchored' else (3, 2, 2))
        with threadpool_limits(limits=1):
            loop.run('fold_promotion', n_permutations=reference['uel']['n_permutations'], prep_each_round=True, progress_bar=False,
                     recording=FULL, record_execution=True, record_model_outputs=True,
                     workers=2 if kind == 'parallel' else 1)
            selected = 3 if kind in ('anchored', 'rolling') else 0
            trainer = Trainer(output, data=bars)
            pid = next(iter(trainer._round_data))
            sensor = trainer.train([pid], **({} if kind == 'ordinary' else {'fold': selected}))[0]
        destination = root / kind
        shutil.move(output, destination)
        metadata = json.loads((destination / 'metadata.json').read_text())
        record = json.loads(_round_storage.text(_round_storage.round_path(destination)).splitlines()[0])
        result[kind] = Campaign(destination, bars, copy.deepcopy(reference), metadata, record, sensor, selected, time.perf_counter() - started)
    (root / 'timings.json').write_text(json.dumps({key: item.elapsed for key, item in result.items()}, indent=2))
    return result
