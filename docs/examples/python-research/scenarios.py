import gzip
import hashlib
import json
import signal
from dataclasses import replace
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from limen import Cohort, Sensor, UniversalExperimentLoop
from limen.backtest import ProductSpec, TradeInputs, TradePolicy, trade_execution
from limen.backtest.execution_events import with_predictions
from limen.backtest.funding_adapter import prepare_funding
from limen.backtest.trade_contract import FundingPolicy, source_binding
from limen.calibration import grid_threshold_optimizer, sklearn_probability_calibrator
from limen.data.utils import compute_data_bars, split_walk_forward
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy, RandomStrategy, SobolStrategy
from limen.experiment.reducer import BudgetReducer
from limen.metrics.balanced_metric import balanced_metric

import research as project
from recorded_inputs import FUNDING, HASHES, funding_history, trade_profile
from cost_research import run_cost_experiment
from limen.backtest.observed_costs import CostCalibration

INPUTS = {
    'spot_1h_20240101_20241231.parquet': 'a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72',
    'spot_15m_20250101_20250531.parquet': 'eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a',
}


def recorded(data_dir, name, rows):
    source = data_dir / name
    if hashlib.sha256(source.read_bytes()).hexdigest() != INPUTS[name]:
        raise ValueError('Recorded input digest differs: ' + name)
    return pl.read_parquet(source, n_rows=rows)


def quarter_hour(data_dir):
    return recorded(data_dir, 'spot_15m_20250101_20250531.parquet', 1920)


def recordable_prep(raw, round_params):
    data = project.prep(raw, round_params)
    validation = data['validation']
    data['_alignment'] = {'missing_datetimes': [], 'first_test_datetime': validation['datetime'][0], 'last_test_datetime': validation['datetime'][-1]}
    return data


# region search
def search(data_dir, output):
    bars = quarter_hour(data_dir)[:1440]
    space = {'window': [4, 12], 'alpha': [0.1, 1.0]}
    runs = {}
    for name in ('grid', 'random', 'sobol', 'shuffled'):
        domain = ParamDomain(space)
        strategy = (GridStrategy(domain, shuffle=True, seed=42) if name == 'shuffled'
                    else {'grid': GridStrategy, 'random': RandomStrategy, 'sobol': SobolStrategy}[name](domain, **({} if name == 'grid' else {'seed': 42})))

        def model(data, round_params):
            fitted = project.fit(data['train'], round_params)
            return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

        loop = UniversalExperimentLoop(data=bars, sfd=SimpleNamespace(params=lambda: space, prep=project.prep, model=model), search_strategy=strategy)
        loop.run(str(output / name), n_permutations=4, workers=1, progress_bar=False)
        runs[name] = project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts()
    return {'runs': runs, 'seed': 42, 'adaptive': adaptive_search(data_dir, output / 'adaptive')}
# endregion search


# region adaptive_search
def adaptive_search(data_dir, output):
    space = {'window': [4, 12], 'alpha': [.1, 1.]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)
    loop = UniversalExperimentLoop(data=quarter_hour(data_dir)[:1440], sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)),
                                   experiment_dir=output, checkpoint_interval=1, feedback_interval=1,
                                   pruning_strategies=[BudgetReducer(max_permutations=3)])
    loop.run('python-adaptive-search', n_permutations=4, workers=1, progress_bar=False)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())['msq_state']
    return {'rows': project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'checkpoint_yielded': checkpoint['yielded_count'], 'checkpoint_remaining': checkpoint['trim_budget']}
# endregion adaptive_search


# region recovery
def recovery(data_dir, output):
    bars = quarter_hour(data_dir)[:1440]
    space = {'window': [4, 12], 'alpha': [0.1, 1.0]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        forecasts = fitted.predict(data['validation'].select('momentum'))
        return {'validation_net_return': project.backtest(data['validation'], forecasts), '_preds': forecasts}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)

    def stop_after_two(log, msq):
        if log.height == 2:
            signal.raise_signal(signal.SIGINT)

    def loop(callback=None):
        return UniversalExperimentLoop(data=bars, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)), experiment_dir=output / 'experiment', checkpoint_interval=1, feedback_interval=1, intra_callback=callback)

    recording: dict[str, bool] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)
    interrupted = loop(stop_after_two)
    interrupted.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording)
    completed_before_resume = project.completed_log(interrupted).height
    resumed = loop()
    resumed.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording, resume=True)
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    return {'interrupted_rounds': completed_before_resume, 'rounds': project.completed_log(resumed).height, 'round_indices': [row['_round_index'] for row in records], 'round_ids': [row['round_id'] for row in records], 'fitted_state_restored': False}
# endregion recovery


# region walk_forward
def walk_forward(data_dir, output):
    bars = quarter_hour(data_dir).head(384)
    results = []
    for anchored in (True, False):
        for index, (train, test) in enumerate(split_walk_forward(bars, n_folds=2, test_bars=48, purge_bars=4, embargo_bars=3, anchored=anchored)):
            fitting = int(train.height * .8)
            candidates = []
            for alpha in (.1, 1.0):
                params = {'window': 4, 'alpha': alpha}
                fitted = project.fit(project.prepare_split(train[:fitting], params), params)
                validation = project.prepare_split(train[fitting:], params)
                score = project.backtest(validation, fitted.predict(validation.select('momentum')))
                candidates.append((score, alpha, fitted))
            score, alpha, fitted = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
            sensor = Sensor.from_model(fitted, prepare=project.prepare_live, params={'window': 4, 'alpha': alpha}, fitting_cutoff=train['datetime'][fitting], selection_cutoff=test['datetime'][0], output_id='next_bar_return')
            history = bars.filter(pl.col('datetime') <= test['datetime'][-1])
            predictions = [point for point in sensor.predict_all(history) if point.datetime >= test['datetime'][0]]
            results.append({'anchored': anchored, 'fold': index, 'train_rows': train.height, 'test_rows': test.height, 'train_end': train['datetime'][-1].isoformat(), 'test_start': test['datetime'][0].isoformat(), 'alpha': alpha, 'validation_net_return': score, 'predictions': [point.prediction for point in predictions], 'test_open': test['open'].to_list(), 'test_close': test['close'].to_list()})
    return {'folds': results, 'ownership': 'caller-owned folds and selection; no recorded compiled fold promotion'}
# endregion walk_forward


# region holdout
def holdout(data_dir, output):
    bars = quarter_hour(data_dir)
    cohort, _, selected, retained = project.run_research(bars, str(output / 'selection'))
    tape = cohort.predict_all(bars[1428:1920])[12:]
    # A decision at the current completed close trades the next open-to-close.
    predictions = []
    for point in tape[:-1]:
        if point.prediction is None:
            raise ValueError('A final assessment decision is unavailable: ' + str(point.reason))
        predictions.append(point.prediction)
    positions = np.sign(np.asarray(predictions, dtype=np.float64))
    next_returns = bars['close'].to_numpy()[1441:1920] / bars['open'].to_numpy()[1441:1920] - 1
    net = positions * next_returns - abs(positions) * .0002
    pl.DataFrame({'decision': [point.datetime for point in tape[:-1]], 'position': positions, 'net_return': net}).write_parquet(output / 'assessment.parquet')
    return {'selected': selected, 'retained_models': len(retained), 'assessment_rows': len(net), 'mean_net_return': float(net.mean()), 'fitting_cutoff': bars['datetime'][960].isoformat(), 'selection_cutoff': bars['datetime'][1440].isoformat(), 'protection': 'caller-owned exclusion; no project reservation or exposure-history guarantee'}
# endregion holdout


def infer_classifier(history, state, params):
    prepared = project.prepare_live(history, params)
    valid = prepared['momentum'].is_not_null().to_numpy()
    probabilities = np.full(prepared.height, np.nan)
    probabilities[valid] = state.model.predict_proba(prepared.filter(pl.col('momentum').is_not_null()).select('momentum'))[:, 1]
    metadata = prepared.select('datetime', 'available_at_ns').with_columns(pl.when(pl.Series(valid)).then(None).otherwise(pl.lit('warm-up')).alias('reason'))
    return metadata, [(int(value >= state.threshold), float(value)) if np.isfinite(value) else (None, None) for value in probabilities]


# region calibration
def calibration(data_dir, output):
    bars = quarter_hour(data_dir)
    params = {'window': 4, 'alpha': .1}
    train, validation = project.prep(bars, params).values()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, random_state=42)).fit(train.select('momentum'), train['next_return'].to_numpy() > 0)
    modes = []
    for calibrate in (False, True):
        fitted = sklearn_probability_calibrator(model, validation.select('momentum'), validation['next_return'].to_numpy() > 0, method='sigmoid') if calibrate else model
        probabilities = fitted.predict_proba(validation.select('momentum'))[:, 1]
        for tune in (False, True):
            threshold, score = grid_threshold_optimizer(validation['next_return'].to_numpy() > 0, probabilities, metric=balanced_metric) if tune else (.5, None)
            state = SimpleNamespace(model=fitted, threshold=threshold)
            sensor = Sensor.from_python(fitted_state=state, params=params, infer=infer_classifier, map_output=lambda rows: rows, state_id=f'sigmoid={calibrate};threshold={tune}', fitting_cutoff=bars['datetime'][960], selection_cutoff=bars['datetime'][1440], output_id='next_bar_up')
            cohort = Cohort.from_python([sensor], aggregate='vote')
            tape = cohort.predict_all(bars[1428:1920])[12:]
            if any(point.prediction not in (0, 1) for point in tape):
                raise ValueError('Classifier inference must provide a binary decision for every assessment row')
            modes.append({'calibrated': calibrate, 'tuned': tune, 'threshold': threshold, 'validation_score': score, 'validation_probabilities': probabilities.tolist(), 'validation_labels': (validation['next_return'].to_numpy() > 0).astype(int).tolist(), 'inference_rows': len(tape), 'positive_predictions': sum(point.prediction == 1 for point in tape)})
    return {'modes': modes, 'validation_reused': True}
# endregion calibration


# region execution
def execution(data_dir, output):
    bars = recorded(data_dir, 'spot_1h_20240101_20241231.parquet', 168)
    interval = 3600 * 1_000_000_000
    observations = bars.with_row_index('row_id').with_columns(pl.col('row_id').cast(pl.String), pl.col('datetime').dt.epoch('ns').alias('start_ns')).with_columns((pl.col('start_ns') + interval).alias('end_ns'), pl.col('start_ns').alias('open_available_at_ns'), (pl.col('start_ns') + interval).alias('available_at_ns')).select('row_id', 'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns', 'open', 'high', 'low', 'close')
    start, end = observations['start_ns'][0], observations['end_ns'][-1]
    signals = observations.select('row_id', 'available_at_ns').with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
    source = source_binding(observations, INPUTS['spot_1h_20240101_20241231.parquet'], start, end, interval, 'recorded hourly OHLC; high/low order unknown')
    inputs = TradeInputs(10000, start, end, signals, observations, None, (source,))
    inputs = with_predictions(inputs, (bars['close'] > bars['open']).cast(pl.Float64).to_numpy())
    policy = TradePolicy(ProductSpec('cash_spot', 'BTCUSDT', 'BTC', 'USDT', .00001, 5), prediction_mode='binary', fee_bps=7, slip_bps=3, execution_lag_seconds=3600, max_holding_seconds=7200)
    ledger = trade_execution(inputs, policy)
    for name in ('states', 'intents', 'fills', 'episodes', 'funding'):
        getattr(ledger, name).write_parquet(output / (name + '.parquet'))
    history = funding_history(data_dir)
    funding = FundingPolicy('limen.backtest.funding_adapter.funding_adapter', 'limen-funding-v1', {}, None, None, 'discrete', 28800.0, 'mark', 'USDT', 'recorded_exact')
    events = prepare_funding(funding, history, start, end)
    binding = source_binding(events, HASHES[FUNDING], start, end, 1_000_000, 'recorded realized rates and mark prices; explicit eight-hour slots')
    funded_inputs = replace(inputs, funding_events=events, sources=(*inputs.sources, binding))
    funded_policy = replace(policy, product=ProductSpec('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0), funding=funding)
    funded = trade_execution(funded_inputs, funded_policy)
    funded.funding.write_parquet(output / 'funding-payments.parquet')
    return {'metrics': dict(ledger.metrics), 'contract_digest': ledger.contract_digest, 'fills': ledger.fills.to_dicts(), 'episodes': ledger.episodes.to_dicts(), 'initial_equity': 10000, 'fee_bps': 7, 'slip_bps': 3, 'source_interval_seconds': 3600, 'funded': {'metrics': dict(funded.metrics), 'payments': funded.funding.to_dicts(), 'recorded_events': history.height, 'input_sha256': HASHES[FUNDING]}}
# endregion execution


# region bars
def bars(data_dir, output):
    raw = recorded(data_dir, 'spot_1h_20240101_20241231.parquet', 384)
    results = []
    for kind, parameter, column in [('volume', 'volume_threshold', 'volume'), ('trade', 'trade_threshold', 'no_of_trades'), ('liquidity', 'liquidity_threshold', 'liquidity_sum')]:
        median = raw[column].median()
        if not isinstance(median, (int, float)):
            raise ValueError('Bar formation requires a numeric recorded threshold source')
        threshold = float(median) * 4
        if kind == 'trade':
            threshold = int(threshold)
        formed = compute_data_bars(raw, bar_type=kind, **{parameter: threshold})
        formed.write_parquet(output / (kind + '.parquet'))
        results.append({'kind': kind, 'threshold': threshold, 'rows': formed.height, 'source_rows_in_complete_bars': int(formed['bar_count'].sum()), 'volume': float(formed['volume'].sum()), 'first_open': formed['open'][0], 'last_close': formed['close'][-1]})
    return {'source_rows': raw.height, 'formed': results, 'incomplete_tail': 'omitted; do not treat it as a completed bar', 'trade_profile': trade_profile(data_dir, output)}
# endregion bars


# region observed_costs
def observed_costs(data_dir, output):
    fixture = data_dir / 'observed_costs'
    expected = {'history.zip': 'df4d7776172df459ec297cbcfe8e72a4748a51ab626e3c990deec83415102093',
                'benchmark.json': 'b71011697713417b21302e670abf78c0bff739ec9135c7012db02fa3b2d6ccf8'}
    for name, digest in expected.items():
        if hashlib.sha256((fixture / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Recorded execution-cost input differs: ' + name)
    with ZipFile(fixture / 'history.zip') as archive:
        history = output / 'history.json'
        history.write_bytes(archive.read('admitted.json'))
    calibration = CostCalibration.from_history(history, cutoff_ns=1785810791000000000)
    artifact, specification = output / 'calibration.json', output / 'research-specification.json'
    calibration.write(artifact)
    specification.write_text(json.dumps({'calibration_digest': calibration.digest}))
    reports = []
    for mode in ('binary', 'target_exposure'):
        for hour in (1, 2):
            experiment, comparison = run_cost_experiment(artifact, specification, fixture / 'benchmark.json',
                                                       output / f'{mode}-{hour}', mode=mode, hour=hour)
            reports.append({'mode': mode, 'hour': hour, 'rounds': project.completed_log(experiment).height,
                            'comparison': comparison.report()})
    return {'calibration': calibration.report(), 'runs': reports,
            'historical_delivery': 'unknown; retrospective stress only'}
# endregion observed_costs


SCENARIOS = {'search': search, 'recovery': recovery, 'walk_forward': walk_forward, 'holdout': holdout, 'calibration': calibration, 'execution': execution, 'bars': bars, 'observed_costs': observed_costs}


def run(name, data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    return SCENARIOS[name](data_dir, output)
