import gzip
import hashlib
import json
import signal
from dataclasses import replace
from datetime import timedelta
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
from limen.backtest.trade_contract import FundingPolicy, export_trade_contract, source_binding
from limen.calibration import grid_threshold_optimizer, sklearn_probability_calibrator
from limen.data.utils import compute_data_bars, split_walk_forward
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy, RandomStrategy, SobolStrategy
from limen.experiment.reducer import BudgetReducer
from limen.metrics.balanced_metric import balanced_metric

import research as project
from recorded_inputs import FUNDING, HASHES, funding_history, trade_profile
from cost_research import run_cost_experiment
from helper_cases import preparation_helpers
from target_cases import label_diagnostics
from analysis_cases import analyze_candidates
from inference_cases import compose_selected
from source_cases import read_source_formats
from metric_cases import evaluate_three_classes
from rule_cases import evaluate_rules
from linked_cases import residual_research
from snapshot_cases import captured_state
from bar_cases import adaptive_trade_bars
from limen.backtest.observed_costs import CostCalibration

INPUTS = {
    'spot_1h_20240101_20241231.parquet': 'a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72',
    'spot_15m_20250101_20250531.parquet': 'eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a',
}


# region recorded_input
def recorded(data_dir, name, rows):
    source = data_dir / name
    if hashlib.sha256(source.read_bytes()).hexdigest() != INPUTS[name]:
        raise ValueError('Recorded input digest differs: ' + name)
    return pl.read_parquet(source, n_rows=rows)
# endregion recorded_input


def quarter_hour(data_dir):
    return recorded(data_dir, 'spot_15m_20250101_20250531.parquet', 1920)


# region hourly_source
def hourly(data_dir, rows):
    return recorded(data_dir, 'spot_1h_20240101_20241231.parquet', rows)


def prepare_hourly_live(history, params):
    return project.features(history, params['window']).with_columns(
        (pl.col('datetime').dt.epoch('ns') + 3_600_000_000_000).alias('available_at_ns'))
# endregion hourly_source


def recordable_prep(raw, round_params):
    data = project.prep(raw, round_params)
    validation = data['validation']
    data['_alignment'] = {'missing_datetimes': [], 'first_test_datetime': validation['datetime'][0], 'last_test_datetime': validation['datetime'][-1]}
    return data


# region preparation
def preparation(data_dir, output):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from limen.features import fractional_diff, find_min_d
    from limen.indicators import roc
    from limen.scalers import RobustScaler
    from limen.targets import TripleBarrierTarget
    from limen.utils import adf_test

    raw = hourly(data_dir, 1440)
    partitions = [raw[:864], raw[864:1152], raw[1152:1440]]
    d = find_min_d(partitions[0], 'close', d_end=.4, step=.2, threshold=.05)
    stationarity = adf_test(fractional_diff(partitions[0], d=d, cols=['close'], threshold=.05)['close_fracdiff'])
    cases = []
    for compressed in (False, True):
        name = 'compressed' if compressed else 'baseline'
        directory = output / name
        directory.mkdir(parents=True)
        target = TripleBarrierTarget(partitions[0], 'label', upper_multiple=1.5,
                                     lower_multiple=1., max_horizon=3, span=8, min_periods=8)
        columns = ['close', 'roc_3', 'roc_12', 'close_fracdiff']
        dropped = ['roc_12'] if compressed else []
        features = [column for column in columns if column not in dropped]
        prepared = []
        for partition in partitions:
            frame = fractional_diff(roc(roc(partition, period=3), period=12),
                                    d=.4, cols=['close'], threshold=.05)
            prepared.append(target.transform(frame).select('datetime', *features, 'label').drop_nulls())
        scaler = RobustScaler(prepared[0].select(features))
        scaled = [scaler.transform(frame.select(features)) for frame in prepared]
        pca = PCA(n_components=2, svd_solver='full') if compressed else None
        matrices = ([pca.fit_transform(scaled[0].to_numpy()),
                     *[pca.transform(frame.to_numpy()) for frame in scaled[1:]]]
                    if pca is not None else [frame.to_numpy() for frame in scaled])
        names = ['pc_0', 'pc_1'] if compressed else features
        model = Ridge(alpha=1., solver='svd').fit(matrices[0], prepared[0]['label'].to_numpy())
        predictions = model.predict(matrices[2])
        for split, frame, matrix in zip(('train', 'val', 'test'), prepared, matrices, strict=True):
            pl.DataFrame(matrix, schema=names, orient='row').with_columns(
                frame['datetime'], frame['label']).write_parquet(directory / (split + '.parquet'))
        cases.append({'name': name, 'features': features, 'dropped': dropped,
                      'rows': [frame.height for frame in prepared],
                      'medians': scaler.medians, 'iqrs': scaler.iqrs,
                      'pca_mean': None if pca is None else pca.mean_.tolist(),
                      'components': None if pca is None else pca.components_.tolist(),
                      'predictions': predictions.tolist(),
                      'rmse': float(np.sqrt(np.mean((predictions-prepared[2]['label'].to_numpy())**2)))})
    return {'cases': cases, 'partition_rows': [864, 288, 288], 'carried_context': False,
            'fixed_d': .4, 'weight_threshold': .05, 'adf_grid_result': d,
            'adf_stationary': bool(stationarity.stationary), 'adf_p_value': float(stationarity.p_value)}
# endregion preparation


# region common_baselines
def common_baselines(data_dir, output):
    from limen.sfd.rule_based.common_strategies import STRATEGY_NAMES, strategy_signals

    raw = hourly(data_dir, None)
    train_rows, val_rows = raw.height * 8 // 11, raw.height // 11
    evaluation = raw[train_rows + val_rows:]
    output.mkdir(parents=True)
    rows = []
    for name in STRATEGY_NAMES:
        signals = strategy_signals(evaluation, name)
        # Caller-owned one-bar trades: enter next open, exit that close, 2 bp total cost.
        net = signals[:-1] * (evaluation['close'].to_numpy()[1:] / evaluation['open'].to_numpy()[1:] - 1 - .0002)
        evaluation.select('datetime').with_columns(pl.Series('desired_position', signals)).write_parquet(output / (name + '.parquet'))
        rows.append({'strategy': name, 'mean_net_return': float(net.mean()),
                     'net': net.tolist(), 'signals': signals.tolist()})
    return {'rows': rows, 'split_rows': [train_rows, val_rows, evaluation.height],
            'round_trip_bps': 2., 'execution': 'independent next-open-to-close trades'}
# endregion common_baselines


# region search
def search(data_dir, output):
    bars = hourly(data_dir, 1440)
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
    return {'runs': runs, 'common_baselines': common_baselines(data_dir, output / 'common-baselines'), 'preparation': preparation(data_dir, output / 'preparation'), 'seed': 42, 'adaptive': adaptive_search(data_dir, output / 'adaptive'),
            'feedback': mixed_feedback(data_dir, output / 'feedback'), 'helper_diagnostics': preparation_helpers(bars, data_dir, output / 'helpers'), 'target_diagnostics': label_diagnostics(bars, data_dir, output / 'targets'), 'analysis': analyze_candidates(bars, output / 'analysis'), 'input_formats': read_source_formats(data_dir), 'multiclass': evaluate_three_classes(bars), 'rules': evaluate_rules(bars), 'linked': residual_research(bars, output / 'linked')}
# endregion search


# region adaptive_search
def adaptive_search(data_dir, output):
    space = {'window': [4, 12], 'alpha': [.1, 1.]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)
    loop = UniversalExperimentLoop(data=hourly(data_dir, 1440), sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)),
                                   experiment_dir=output, checkpoint_interval=1, feedback_interval=1,
                                   pruning_strategies=[BudgetReducer(max_permutations=3)])
    loop.run('python-adaptive-search', n_permutations=4, workers=1, progress_bar=False)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())['msq_state']
    return {'rows': project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'checkpoint_yielded': checkpoint['yielded_count'], 'checkpoint_remaining': checkpoint['trim_budget']}
# endregion adaptive_search


# region mixed_feedback
def mixed_feedback(data_dir, output):
    space = {'window': [4, 12], 'alpha': [.1, 1.]}

    def steer(log, msq):
        if log.height == 2:
            msq.keep_is('alpha', .1)
            (output / 'interventions.json').write_text(json.dumps([
                {'op': 'inject_value', 'param': 'alpha', 'value': 2.0}]))

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        return {'validation_net_return': project.backtest(data['validation'], fitted.predict(data['validation'].select('momentum')))}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)
    loop = UniversalExperimentLoop(data=hourly(data_dir, 1440), sfd=sfd,
                                   search_strategy=GridStrategy(ParamDomain(space)), experiment_dir=output,
                                   checkpoint_interval=1, feedback_interval=1, intra_callback=steer,
                                   pruning_strategies=[BudgetReducer(max_permutations=3)])
    loop.run('python-feedback', n_permutations=4, workers=1, progress_bar=False)
    checkpoint = json.loads((output / 'checkpoint.json').read_text())
    return {'rows': project.completed_log(loop).select('window', 'alpha', 'validation_net_return').to_dicts(),
            'audit': [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()],
            'domain': checkpoint['domain_state'], 'yielded': checkpoint['msq_state']['yielded_count'],
            'remaining': checkpoint['msq_state']['trim_budget']}
# endregion mixed_feedback


# region recovery
def recovery(data_dir, output):
    bars = hourly(data_dir, 1440)
    space = {'window': [4, 12], 'alpha': [0.1, 1.0]}

    def model(data, round_params):
        fitted = project.fit(data['train'], round_params)
        forecasts = fitted.predict(data['validation'].select('momentum'))
        return {'validation_net_return': project.backtest(data['validation'], forecasts), '_preds': forecasts}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: space, prep=recordable_prep, model=model)

    def stop_after_two(log, msq):
        if log.height == 2:
            signal.raise_signal(signal.SIGINT)

    def loop(callback=None, directory='experiment'):
        return UniversalExperimentLoop(data=bars, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)), experiment_dir=output / directory, checkpoint_interval=1, feedback_interval=1, intra_callback=callback)

    recording: dict[str, bool] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)
    interrupted = loop(stop_after_two)
    interrupted.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording)
    completed_before_resume = project.completed_log(interrupted).height
    resumed = loop()
    resumed.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording, resume=True)
    with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
        records = [json.loads(line) for line in stream]
    full = loop(directory='uninterrupted')
    full.run('python-recovery', n_permutations=4, workers=1, progress_bar=False, recording=recording)
    if not project.completed_log(resumed).drop('execution_time').equals(project.completed_log(full).drop('execution_time')):
        raise ValueError('Resumed Python results differ from uninterrupted research')
    return {'interrupted_rounds': completed_before_resume, 'rounds': project.completed_log(resumed).height, 'round_indices': [row['_round_index'] for row in records], 'round_ids': [row['round_id'] for row in records], 'fitted_state_restored': False, 'uninterrupted_equivalence': True,
            'captured': captured_state(bars, output / 'captured')}
# endregion recovery


# region walk_forward
def walk_forward(data_dir, output):
    bars = hourly(data_dir, 384)
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
            sensor = Sensor.from_model(fitted, prepare=prepare_hourly_live, params={'window': 4, 'alpha': alpha}, fitting_cutoff=train['datetime'][fitting], selection_cutoff=test['datetime'][0], output_id='next_bar_return')
            history = bars.filter(pl.col('datetime') <= test['datetime'][-1])
            predictions = [point for point in sensor.predict_all(history) if point.datetime >= test['datetime'][0]]
            results.append({'anchored': anchored, 'fold': index, 'train_rows': train.height, 'test_rows': test.height, 'train_end': train['datetime'][-1].isoformat(), 'test_start': test['datetime'][0].isoformat(), 'alpha': alpha, 'validation_net_return': score, 'predictions': [point.prediction for point in predictions], 'available_at_ns': [point.available_at_ns for point in predictions], 'test_open': test['open'].to_list(), 'test_close': test['close'].to_list()})
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
    if valid.any():
        probabilities[valid] = state.model.predict_proba(prepared.filter(pl.col('momentum').is_not_null()).select('momentum'))[:, 1]
    metadata = prepared.select('datetime', 'available_at_ns').with_columns(pl.when(pl.Series(valid)).then(None).otherwise(pl.lit('warm-up')).alias('reason'))
    return metadata, [(int(value >= state.threshold), float(value)) if np.isfinite(value) else (None, None) for value in probabilities]


# region calibration
def calibration(data_dir, output):
    bars = quarter_hour(data_dir)
    params = {'window': 4, 'alpha': .1}
    train, validation = project.prep(bars, params).values()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, random_state=42)).fit(train.select('momentum'), train['next_return'].to_numpy() > 0)
    modes, sensors = [], []
    for calibrate in (False, True):
        fitted = sklearn_probability_calibrator(model, validation.select('momentum'), validation['next_return'].to_numpy() > 0, method='sigmoid') if calibrate else model
        probabilities = fitted.predict_proba(validation.select('momentum'))[:, 1]
        for tune in (False, True):
            threshold, score = grid_threshold_optimizer(validation['next_return'].to_numpy() > 0, probabilities, metric=balanced_metric) if tune else (.5, None)
            state = SimpleNamespace(model=fitted, threshold=threshold)
            sensor = Sensor.from_python(fitted_state=state, params=params, infer=infer_classifier, map_output=lambda rows: rows, state_id=f'sigmoid={calibrate};threshold={tune}', fitting_cutoff=bars['datetime'][960], selection_cutoff=bars['datetime'][1440], output_id='next_bar_up')
            sensors.append(sensor)
            cohort = Cohort.from_python([sensor], aggregate='vote')
            tape = cohort.predict_all(bars[1428:1920])[12:]
            if any(point.prediction not in (0, 1) for point in tape):
                raise ValueError('Classifier inference must provide a binary decision for every assessment row')
            modes.append({'calibrated': calibrate, 'tuned': tune, 'threshold': threshold, 'validation_score': score, 'validation_probabilities': probabilities.tolist(), 'validation_labels': (validation['next_return'].to_numpy() > 0).astype(int).tolist(), 'inference_rows': len(tape), 'positive_predictions': sum(point.prediction == 1 for point in tape)})
    methods = []
    for method in ('sigmoid', 'isotonic'):
        calibrated = sklearn_probability_calibrator(model, validation.select('momentum'), validation['next_return'].to_numpy() > 0, method=method)
        probabilities = calibrated.predict_proba(validation.select('momentum'))[:, 1]
        threshold, score = grid_threshold_optimizer(validation['next_return'].to_numpy() > 0, probabilities, metric=balanced_metric)
        state = SimpleNamespace(model=calibrated, threshold=threshold)
        sensor = Sensor.from_python(fitted_state=state, params=params, infer=infer_classifier, map_output=lambda rows: rows,
            state_id=method, fitting_cutoff=bars['datetime'][960], selection_cutoff=bars['datetime'][1440], output_id='next_bar_up')
        tape = sensor.predict_all(bars[1428:1920])[12:]
        methods.append({'method': method, 'calibrated': True, 'tuned': True, 'threshold': threshold, 'validation_score': score,
            'validation_probabilities': probabilities.tolist(), 'validation_labels': (validation['next_return'].to_numpy() > 0).astype(int).tolist(),
            'inference_rows': len(tape), 'positive_predictions': sum(point.prediction == 1 for point in tape)})
    return {'probability_methods': methods, 'modes': modes, 'decisions': [{key: mode[key] for key in ('calibrated', 'tuned', 'threshold', 'inference_rows', 'positive_predictions')} for mode in modes], 'validation_reused': True, 'composition': compose_selected(sensors, bars)}
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
    for name in ('states', 'intents', 'fills', 'episodes', 'funding'):
        getattr(funded, name).write_parquet(output / ('funded-' + name + '.parquet'))
    context = {'observations': observations.to_dicts(), 'partition': [start, end], 'contract': export_trade_contract(policy, inputs)}
    funded_context = context | {'contract': export_trade_contract(funded_policy, funded_inputs)}
    return {'metrics': dict(ledger.metrics), 'contract_digest': ledger.contract_digest, 'fills': ledger.fills.to_dicts(), 'ledger_excerpt': ledger.fills.head(8).to_dicts(), 'episodes': ledger.episodes.to_dicts(), 'context': context, 'initial_equity': 10000, 'fee_bps': 7, 'slip_bps': 3, 'source_interval_seconds': 3600, 'funded': {'metrics': dict(funded.metrics), 'contract_digest': funded.contract_digest, 'payments': funded.funding.to_dicts(), 'fills': funded.fills.to_dicts(), 'context': funded_context, 'recorded_events': history.height, 'input_sha256': HASHES[FUNDING]}}
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
        results.append({'kind': kind, 'threshold': threshold, 'rows': formed.height, 'source_rows_in_complete_bars': int(formed['bar_count'].sum()), 'volume': float(formed['volume'].sum()), 'first_open': formed['open'][0], 'last_close': formed['close'][-1], 'first_start': formed['datetime'][0].isoformat(), 'last_start': formed['datetime'][-1].isoformat(), 'last_available': (raw['datetime'][int(formed['bar_count'].sum()) - 1] + timedelta(hours=1)).isoformat()})
    return {'source_rows': raw.height, 'formed': results, 'incomplete_tail': 'omitted; do not treat it as a completed bar', 'trade_profile': trade_profile(data_dir, output),
            'adaptive': adaptive_trade_bars(data_dir, output / 'adaptive')}
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
    result = SCENARIOS[name](data_dir, output)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
