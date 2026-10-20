"""Four frozen source-independent CLI manifests and baseline recorder."""
from __future__ import annotations
from limen.experiment import _round_storage as _storage
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import polars as pl
from ruamel.yaml import YAML
from limen.cli.commands.run import run_experiment
from limen.data import HistoricalData
from limen.experiment.manifest_core import DataSourceConfig, DataSourceResolver

NUMERICAL_ENV = {
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'OPENBLAS_CORETYPE': 'NEHALEM',
    'NPY_DISABLE_CPU_FEATURES': 'SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2,AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR',
}

ROOT = Path(__file__).resolve().parents[3]
MARKET = ROOT / 'tests/fixtures/dollar_bar_crash_reversal_15m.parquet'
BASELINE = '7746615031a9a2af30905bc218977b624f718143'
CASES = ('binary_costs','directional_barriers','interleaved','rule_based')

def fixture_source(config: DataSourceConfig) -> pl.DataFrame:
    assert config.method is HistoricalData.get_spot_klines
    return pl.read_parquet(MARKET)

def manifest(case: str) -> dict:
    dates = {'train_start':'2026-01-01','train_end':'2026-01-22','val_start':'2026-01-23',
             'val_end':'2026-01-31','test_start':'2026-02-01','test_end':'2026-02-07'}
    base = {'data_source':{'method':'limen.data.HistoricalData.get_spot_klines',
                           'params':{'kline_size':3600}},'split_dates':dates}
    if case == 'binary_costs':
        m = {**base,'type':'ml',
             'indicators':[{'func':'limen.indicators.window_return','params':{'period':1}}],
             'features':[{'func':'limen.features.lag_range','params':{'col':'ret_1','start':0,'end':8}}],
             'target':{'name':'up_next','class':'limen.targets.QuantileBinaryTarget',
                       'fit_params':{'source_column':'ret_1','quantile':0.5},
                       'transform_params':{'shift':-1}},
             'reference_architecture':'limen.sfd.reference_architecture.lightgbm_binary.lightgbm_binary',
             'backtest':{'fee_bps':'{fee}','slip_bps':'{slip}','notional_rate':'{size}'}}
        p = {'fee':[3.,12.],'slip':[2.,7.],'size':[0.25,1.0], 'n_jobs':[1],
             'random_state':[42],'deterministic':[True],'force_row_wise':[True],
             'n_estimators':[20],'early_stopping_rounds':[0],'subsample_freq':[0],
             'subsample':[1.0],'colsample_bytree':[1.0]}
    elif case in ('directional_barriers','interleaved'):
        m = {**base,'type':'ml',
             'indicators':[{'func':'limen.indicators.window_return','params':{'period':1}}],
             'features':[{'func':'limen.features.lag_range',
                          'params':{'col':'ret_1','start':0,'end':'{lookback_end}'}}],
             'target':{'name':'next_return','class':'limen.targets.NextReturnTarget',
                       'transform_params':{'periods':1,'scale':100.}},
             'reference_architecture':'limen.sfd.reference_architecture.dlinear_regressor.dlinear_regressor',
             'backtest':{'fee_bps':'{fee}','slip_bps':5.,'notional_rate':1.,
                         'take_profit_bps':'{tp}','stop_loss_bps':'{sl}'}}
        if case == 'directional_barriers':
            p={'lookback_end':[13],'kernel_size':[13],'alpha':[1.],
               'fee':[3.,10.],'tp':[None,5.],'sl':[None,5.]}
        else:
            p={'lookback_end':[13,23],'kernel_size':[13],'alpha':[1.,20.],
               'fee':[3.,10.],'tp':[None,5.],'sl':[5.]}
    elif case == 'rule_based':
        m = {**base,'type':'rule_based',
             'strategy':{'conditions':[{'id':'enter','name':'up bar','type':'relative',
                                        'column':'close','operator':'>','other_column':'open'}],
                         'entry':'enter'},
             'reference_architecture':'limen.sfd.reference_architecture.rule_based.rule_based',
             'backtest':{'fee_bps':'{fee}','slip_bps':5.,'notional_rate':'{size}',
                         'take_profit_bps':'{tp}','stop_loss_bps':'{sl}'}}
        p={'fee':[3.,10.],'size':[0.5,1.],'tp':[None,5.],'sl':[None,5.]}
    else:
        raise ValueError(case)
    return {'schema_version':'1.0',
            'metadata':{'name':case,'limen_version':'5.21.0','mode':'development',
                        'description':'Frozen factorization comparison'},
            'sfd':{'manifest':m,'params':p},
            'uel':{'n_permutations':math.prod(len(v) for v in p.values()),
                   'search_strategy':{'type':'grid'},'prep_each_round':True,
                   'record_execution':True,'record_model_outputs':case=='binary_costs',
                   'feedback_interval':2,'output_path':case}}

def execute(case: str, directory: Path, factorize: bool | None = None) -> dict:
    cfg=manifest(case)
    cfg['uel']['recording'] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)
    if factorize is not None:
        cfg['uel']['factorize']=factorize
    directory.mkdir(parents=True,exist_ok=True)
    yaml=directory/'manifest.yaml'
    with yaml.open('w') as stream:
        YAML().dump(cfg,stream)
    with patch.object(DataSourceResolver,'resolve',staticmethod(fixture_source)):
        assert run_experiment(yaml,results_base=directory,progress_bar=False)
    output=directory/'results'/'dev'/case
    with (output/'results.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    return {'columns':list(rows[0]) if rows else [],'rows':rows,
            'round_data':[json.loads(s) for s in _storage.text(_storage.round_path(output)).splitlines()],
            'metadata':json.loads((output/'metadata.json').read_text())}

def _scalar(value: float | str, in_csv: bool) -> object:
    if type(value) is float:
        if math.isnan(value):
            return ['nan']
        if math.isinf(value):
            return ['infinity', math.copysign(1., value)]
        return ['float', format(value, '.12f'), math.copysign(1., value) < 0 and value == 0]
    if isinstance(value, str):
        if in_csv and value.lower() in ('true', 'false'):
            return ['bool', value.lower() == 'true']
        if in_csv and value:
            try:
                return canonical(float(value))
            except ValueError:
                pass
        return ['str', value]
    raise TypeError(type(value).__name__)


def canonical(value: object, *, in_csv: bool = False) -> object:
    """Strict types and null positions; decimal-normalize floats only."""
    if value is None:
        return ['null']
    if type(value) is bool:
        return ['bool', value]
    if type(value) is int:
        return ['int', value]
    if isinstance(value, (float, str)):
        return _scalar(value, in_csv)
    if isinstance(value, (tuple, list)):
        return [canonical(x, in_csv=in_csv) for x in value]
    if isinstance(value, dict):
        return [[str(k), canonical(v, in_csv=in_csv)] for k,v in value.items()]
    raise TypeError(type(value).__name__)


def baseline_surface(captured: dict) -> dict:
    # Keep the original comparison columns; these independent additions have separate tests.
    return {**captured,
            'columns': [key for key in captured['columns'] if key != '_convergence_warning'],
            'metadata': {key: value for key, value in captured['metadata'].items() if key != 'recording'},
            'rows': [{key: value for key, value in row.items() if key != '_convergence_warning'}
                     for row in captured['rows']]}


def stable(captured: dict) -> dict:
    excluded = {'created_at', 'round_storage', 'limen_version', 'manifest_id', 'yaml_reference', 'factorize'}
    return {'columns': captured['columns'],
            'rows': [{k:v for k,v in row.items() if k!='execution_time'} for row in captured['rows']],
            'round_data':captured['round_data'],
            'metadata':{k:v for k,v in captured['metadata'].items() if k not in excluded}}


def scientific(captured: dict) -> dict:
    science = stable(captured)
    # CSV is inherently textual: normalize only numeric cell representations.
    return {'columns':science['columns'],
                 'rows':[canonical(row,in_csv=True) for row in science['rows']],
                 'round_data':canonical(science['round_data']),
                 'metadata':canonical(science['metadata'])}
def _hash(value: object) -> str:
    payload=json.dumps(value,separators=(',',':'),ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def digest(captured: dict) -> str:
    return _hash(scientific(captured))


def component_digests(captured: dict) -> dict[str, str]:
    science = stable(captured)
    return {'columns': _hash(science['columns']),
            **{f'rows/{key}': _hash([canonical(row[key], in_csv=True) for row in science['rows']])
               for key in science['rows'][0]},
            **{f'round_data/{key}': _hash([canonical(record[key]) for record in science['round_data']])
               for key in science['round_data'][0]},
            **{f'metadata/{key}': _hash(canonical(value)) for key, value in science['metadata'].items()}}


def pinned_captures(directory: Path) -> dict:
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--capture', '--out', str(directory)],
                   env={**os.environ, **NUMERICAL_ENV}, check=True)
    return {name: json.loads((directory / f'{name}.json').read_text()) for name in CASES}


def main():
    # Restart before numerical imports when recording a baseline directly.
    if platform.system() == 'Linux' and platform.machine() == 'x86_64' and any(
        os.environ.get(key) != value for key, value in NUMERICAL_ENV.items()
    ):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                       env={**os.environ, **NUMERICAL_ENV}, check=True)
        return
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path(__file__).parent)
    ap.add_argument('--capture', action='store_true')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for name in CASES:
        if args.capture:
            captured = {mode: execute(name, args.out / name / mode, factorize=enabled)
                        for mode, enabled in (('off', None), ('on', True))}
            (args.out / f'{name}.json').write_text(json.dumps(captured))
            continue
        with tempfile.TemporaryDirectory() as temp:
            result=baseline_surface(execute(name,Path(temp)))
        evidence={'source_sha':BASELINE,'case':name,
                  'market_sha256':hashlib.sha256(MARKET.read_bytes()).hexdigest(),
                  'python':platform.python_version(),'platform':platform.platform(),
                  'lock_file':'requirements/ci/research-env.txt',
                  'threads':{k:os.environ.get(k) for k in
                             NUMERICAL_ENV},
                  'digest_12dp':digest(result),
                  'round_ids':[row['round_id'] for row in result['round_data']],
                  'columns':result['columns'],'row_count':len(result['rows']),
                  'component_digests': component_digests(result)}
        dest=args.out/f'{name}.json'
        dest.write_text(json.dumps(evidence,indent=2,default=str)+'\n')
if __name__=='__main__':
    main()
