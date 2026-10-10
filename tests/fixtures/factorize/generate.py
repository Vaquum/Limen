"""Four frozen source-independent CLI manifests and baseline recorder."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from unittest.mock import patch
import polars as pl
from ruamel.yaml import YAML
from limen.cli.commands.run import run_experiment
from limen.data import HistoricalData

ROOT = Path(__file__).resolve().parents[3]
MARKET = ROOT / 'tests/fixtures/dollar_bar_crash_reversal_15m.parquet'
BASELINE = 'cda37a02d526c72360c56da4c75dc0626d08cd4c'
CASES = ('binary_costs','directional_barriers','interleaved','rule_based')

def fixture_source(**_kwargs: object) -> pl.DataFrame:
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
    if factorize is not None:
        cfg['uel']['factorize']=factorize
    directory.mkdir(parents=True,exist_ok=True)
    yaml=directory/'manifest.yaml'
    with yaml.open('w') as stream:
        YAML().dump(cfg,stream)
    with patch.object(HistoricalData,'get_spot_klines',staticmethod(fixture_source)):
        assert run_experiment(yaml,results_base=directory,progress_bar=False)
    output=directory/'results'/'dev'/case
    with (output/'results.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    return {'columns':list(rows[0]) if rows else [],'rows':rows,
            'round_data':[json.loads(s) for s in (output/'round_data.jsonl').read_text().splitlines()],
            'metadata':json.loads((output/'metadata.json').read_text())}

def canonical(value: object, *, in_csv: bool = False) -> object:  # noqa: PLR0911
    """Strict types and null positions; decimal-normalize floats only."""
    if value is None:
        return ['null']
    if type(value) is bool:
        return ['bool', value]
    if type(value) is int:
        return ['int', value]
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
    if isinstance(value, (tuple, list)):
        return [canonical(x, in_csv=in_csv) for x in value]
    if isinstance(value, dict):
        return [[str(k), canonical(v, in_csv=in_csv)] for k,v in value.items()]
    raise TypeError(type(value).__name__)


def stable(captured: dict) -> dict:
    excluded = {'created_at', 'limen_version', 'manifest_id', 'yaml_reference', 'factorize'}
    return {'columns': captured['columns'],
            'rows': [{k:v for k,v in row.items() if k!='execution_time'} for row in captured['rows']],
            'round_data':captured['round_data'],
            'metadata':{k:v for k,v in captured['metadata'].items() if k not in excluded}}


def digest(captured: dict) -> str:
    science = stable(captured)
    # CSV is inherently textual: normalize only numeric cell representations.
    converted = {'columns':science['columns'],
                 'rows':[canonical(row,in_csv=True) for row in science['rows']],
                 'round_data':canonical(science['round_data']),
                 'metadata':canonical(science['metadata'])}
    payload=json.dumps(converted,separators=(',',':'),ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path(__file__).parent)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for name in CASES:
        with tempfile.TemporaryDirectory() as temp:
            result=execute(name,Path(temp))
        evidence={'source_sha':BASELINE,'case':name,
                  'market_sha256':hashlib.sha256(MARKET.read_bytes()).hexdigest(),
                  'python':platform.python_version(),'platform':platform.platform(),
                  'lock_file':'requirements/ci/research-env.txt',
                  'threads':{k:os.environ.get(k) for k in
                             ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                  'digest_12dp':digest(result),
                  'round_ids':[row['round_id'] for row in result['round_data']],
                  'columns':result['columns'],'row_count':len(result['rows'])}
        dest=args.out/f'{name}.json'
        dest.write_text(json.dumps(evidence,indent=2,default=str))
if __name__=='__main__':
    main()
