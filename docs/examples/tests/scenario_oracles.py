from datetime import datetime, timezone, timedelta
from itertools import combinations
from statistics import NormalDist
import hashlib
import json
import math
import subprocess
import gzip

import numpy as np
import polars as pl
import pytest


def ridge_score(bars, window, alpha):
    close, opening = bars['close'].to_numpy(), bars['open'].to_numpy()
    x = close[12:959] / close[12-window:959-window] - 1
    y = close[13:960] / opening[13:960] - 1
    z = (x - x.mean()) / x.std()
    coefficient = np.dot(z, y-y.mean()) / (np.dot(z, z) + alpha)
    val = close[972:1439] / close[972-window:1439-window] - 1
    forecasts = (val-x.mean()) / x.std() * coefficient + y.mean()
    positions = np.sign(forecasts)
    return float(np.mean(positions*(close[973:1440]/opening[973:1440]-1)-abs(positions)*.0002))


def fold_ridge(bars, fold, alpha):
    test_start = 288 + fold['fold'] * 48
    training = bars[(0 if fold['anchored'] else fold['fold']*48):test_start-4]
    fitting = int(training.height * .8)
    close, opening = training['close'].to_numpy(), training['open'].to_numpy()
    x = close[12:fitting-1]/close[8:fitting-5]-1
    y = close[13:fitting]/opening[13:fitting]-1
    z = (x-x.mean())/x.std()
    coefficient = np.dot(z, y-y.mean())/(np.dot(z, z)+alpha)
    vc, vo = close[fitting:], opening[fitting:]
    forecasts = ((vc[12:-1]/vc[8:-5]-1)-x.mean())/x.std()*coefficient+y.mean()
    validation = float(np.mean(np.sign(forecasts)*(vc[13:]/vo[13:]-1)-abs(np.sign(forecasts))*.0002))
    prices = bars['close'].to_numpy()
    inference = ((prices[test_start:test_start+48]/prices[test_start-4:test_start+44]-1)-x.mean())/x.std()*coefficient+y.mean()
    return validation, inference


def execution_clocks(context, ledger, raw):
    source = dict(zip(raw['datetime'].dt.epoch('ns').to_list(), raw.to_dicts(), strict=True))
    observations = {row['row_id']: row for row in context['observations']}
    for row in observations.values():
        assert row['end_ns'] == row['start_ns'] + 3_600_000_000_000
        assert row['available_at_ns'] == row['end_ns'] and row['open_available_at_ns'] == row['start_ns']
        assert all(row[name] == source[row['start_ns']][name] for name in ('open', 'high', 'low', 'close'))
    intents = {row['intent_id']: row for row in ledger['intents']}
    lag = round(context['contract']['policy']['execution_lag_seconds'] * 1_000_000_000)
    for fill in ledger['fills']:
        row, intent = observations[fill['evidence_row_id']], intents[fill['intent_id']]
        assert context['partition'][0] <= fill['time_ns'] <= context['partition'][1]
        eligible = intent['requested_at_ns'] + (lag if intent['reason'] == 'signal' else 0)
        assert fill['time_ns'] >= eligible, 'Fill precedes its declared execution lag'
        assert fill['time_ns'] in (row['start_ns'], row['end_ns'])
        if fill['time_ns'] == row['start_ns']:
            assert fill['reference_price'] == row['open']
        elif intent['reason'] in ('stop_loss', 'take_profit'):
            assert row['low'] <= fill['reference_price'] <= row['high']
        else:
            assert fill['reference_price'] == row['close']


def verify_feedback(interface, feedback, inputs):
    bars = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
    assert len(feedback['rows']) == feedback['yielded'] == 3 and feedback['remaining'] == 0
    assert all(row['errors'] == [] and row['suggestions'] == [] for row in feedback['audit'])
    first = feedback['audit'][0]['interventions']
    assert [item['source'] for item in first] == ['BudgetReducer', 'intra_callback', 'intervention_file']
    assert [item.get('op', item.get('operation')) for item in first] == ['trim', 'keep_is', 'inject_value']
    assert sum(item['source'] == 'intervention_file' for cycle in feedback['audit'] for item in cycle['interventions']) == 1
    assert first[0]['target_count'] == 3
    parameter, injected = ('alpha', 2.) if interface == 'python' else ('C', 7.5)
    assert first[1]['param'] == first[2]['param'] == parameter
    assert first[1]['value'] == .1 and first[2]['value'] == injected
    assert feedback['domain'][parameter] == [.1, injected]
    assert feedback['rows'][-1][parameter] == injected
    if interface == 'python':
        assert [(row['window'], row['alpha']) for row in feedback['rows']] == [(4, .1), (4, 1.), (4, 2.)]
        for row in feedback['rows']:
            assert row['validation_net_return'] == pytest.approx(ridge_score(bars, row['window'], row['alpha']), rel=1e-10, abs=1e-12)
    else:
        assert [row['C'] for row in feedback['rows']] == [.1, .5, 7.5]
        training = bars.filter(pl.col('datetime') < datetime(2024, 1, 15, tzinfo=timezone.utc))['close'].to_numpy()
        cutoff = float(np.median((training[1:] / training[:-1] - 1) * 100))
        labels = bars.with_columns((((pl.col('close').shift(-1) / pl.col('close') - 1) * 100) > cutoff).cast(pl.UInt8).alias('label'))
        feedback_records = {record['round_id']: record for record in feedback['records']}
        assert set(feedback_records) == set(feedback['predictions']) == {row['id'] for row in feedback['rows']}
        for row in feedback['rows']:
            alignment = feedback_records[row['id']]['alignment']
            expected = labels.filter(pl.col('datetime').is_between(datetime.fromisoformat(alignment['first_test_datetime']), datetime.fromisoformat(alignment['last_test_datetime'])))['label'].to_numpy()
            predictions = np.asarray(feedback['predictions'][row['id']])
            assert predictions.size == expected.size > 0
            assert row['accuracy'] == pytest.approx(float(np.mean(predictions == expected)))


def verify_acceptance(result, output, inputs):
    report = result['report']
    assert result['rounds'] == report['n_trials'] >= 2
    for name, digest in result['artifacts'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert json.loads((output / 'acceptance_report.json').read_text()) == report
    frame = pl.read_parquet(output / 'trial_returns.parquet')
    trials = frame['trial'].unique(maintain_order=True).to_list()
    tracks = []
    with gzip.open(output / 'round_data.jsonl.gz', 'rt') as stream:
        records = {item['round_id']: item for item in map(json.loads, stream)}
    source = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet')
    assert source.height == 8784 and set(records) == set(trials)
    dates_by_trial = []
    for trial in trials:
        rows = frame.filter(pl.col('trial') == trial).sort('bar')
        assert rows['bar'].to_list() == list(range(96))
        track = rows['net_return'].to_numpy()
        folds = records[trial]['folds']
        assert len(folds) == 2
        np.testing.assert_array_equal(track, [value for fold in folds for value in fold['net_returns']])
        dates = [value for fold in folds for value in fold['alignment']['test_datetimes']]
        assert len(dates) == len(set(dates)) == 96
        assert dates == sorted(dates)
        assert set(dates) <= {value.isoformat() for value in source['datetime']}
        dates_by_trial.append(dates)
        tracks.append(track)
    assert dates_by_trial[0] == dates_by_trial[1]
    matrix = np.stack(tracks)
    sharpes = matrix.mean(axis=1) / matrix.std(axis=1, ddof=1)
    winner = int(np.argmax(sharpes))
    variance = float(sharpes.var(ddof=1))
    track = matrix[winner]
    centered = track - track.mean()
    standardized = centered / np.sqrt(np.mean(centered ** 2))
    skew, kurtosis = float(np.mean(standardized ** 3)), float(np.mean(standardized ** 4))
    normal, gamma = NormalDist(), 0.5772156649015329
    benchmark = math.sqrt(variance) * ((1-gamma)*normal.inv_cdf(1-1/len(trials)) + gamma*normal.inv_cdf(1-1/(len(trials)*math.e)))
    corrected = 1-skew*sharpes[winner]+(kurtosis-1)*sharpes[winner]**2/4
    probability = normal.cdf((sharpes[winner]-benchmark)*math.sqrt((len(track)-1)/corrected))
    blocks = np.split(np.arange(matrix.shape[1]), 2)
    losses = []
    for selected in combinations(range(2), 1):
        training = matrix[:, np.concatenate([blocks[index] for index in selected])]
        held = matrix[:, np.concatenate([blocks[index] for index in range(2) if index not in selected])]
        chosen = int(np.argmax(training.mean(axis=1) / training.std(axis=1, ddof=1)))
        scores = held.mean(axis=1) / held.std(axis=1, ddof=1)
        rank = np.count_nonzero(scores < scores[chosen]) + (np.count_nonzero(scores == scores[chosen])+1)/2
        losses.append(rank/(len(trials)+1) <= .5)
    pbo = sum(losses)/len(losses)
    assert report['n_bars'] == 96 and report['n_blocks'] == 2 and report['pbo_block_bars'] == 48
    assert report['pbo_combinations'] == 2 and report['pbo_resolution'] == .5
    assert report['winner_trial'] == trials[winner]
    assert report['winner_sharpe'] == pytest.approx(float(sharpes[winner]), abs=1e-14)
    assert report['trial_sharpe_variance'] == pytest.approx(variance, abs=1e-14)
    assert report['deflated_sharpe_probability'] == pytest.approx(probability, abs=1e-14)
    assert report['pbo'] == pbo and report['errors'] == {}
    assert report['thresholds'] == {'min_deflated_sharpe_probability': .95, 'max_pbo': .2}
    assert report['verdicts'] == {'min_deflated_sharpe_probability': probability >= .95, 'max_pbo': pbo <= .2}
    markdown = (output / 'acceptance_report.md').read_text()
    assert f'Deflated Sharpe probability: {report["deflated_sharpe_probability"]}' in markdown
    assert f'PBO: {report["pbo"]}' in markdown
    for name, passed in report['verdicts'].items():
        assert f'{name} ({report["thresholds"][name]}): {"pass" if passed else "fail"}' in markdown


def verify(interface, name, result, output, inputs):
    if name == 'search':
        if interface == 'python':
            from helper_oracles import verify_helpers, verify_targets, verify_multiclass, verify_rules
            verify_helpers(result['helper_diagnostics'], output / 'helpers', inputs)
            verify_targets(result['target_diagnostics'], output / 'targets', inputs)
            verify_multiclass(result['multiclass'], inputs)
            verify_rules(result['rules'], inputs)
            verify_analysis(result['analysis'], inputs)
            verify_source_formats(result['input_formats'], inputs)
        verify_preparation(interface, result['preparation'], output / 'preparation', inputs)
        verify_common_baselines(interface, result['common_baselines'], output / 'common-baselines', inputs)
        expected = {'grid', 'random', 'sobol', 'shuffled'} | ({'factorized'} if interface == 'compiled' else set())
        assert set(result['runs']) == expected
        identities = []
        bars = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
        for strategy, rows in result['runs'].items():
            assert len(rows) == 4
            if interface == 'python':
                assert {(row['window'], row['alpha']) for row in rows} == {(w, a) for w in (4, 12) for a in (.1, 1.)}
                for row in rows:
                    assert row['validation_net_return'] == pytest.approx(ridge_score(bars, row['window'], row['alpha']), rel=1e-10, abs=1e-12)
            else:
                if strategy == 'factorized':
                    assert {(row['n_estimators'], row['fee']) for row in rows} == {(trees, fee) for trees in (4, 8) for fee in (7., 9.)}
                else:
                    assert {row['C'] for row in rows} == {.1, .5, 1., 2.}
                    identities.append({row['C']: row['id'] for row in rows})
        if identities:
            assert all(identity == identities[0] for identity in identities)
        adaptive = result['adaptive']
        assert len(adaptive['rows']) == adaptive['checkpoint_yielded'] == 3 and adaptive['checkpoint_remaining'] == 0
        assert all(row['errors'] == [] for row in adaptive['audit'])
        assert any(item['op'] == 'trim' and item['target_count'] == 3 for row in adaptive['audit'] for item in row['interventions'])
        for row in adaptive['rows']:
            if interface == 'python':
                assert row['validation_net_return'] == pytest.approx(ridge_score(bars, row['window'], row['alpha']), rel=1e-10, abs=1e-12)
            else:
                original = next(item for item in result['runs']['grid'] if item['C'] == row['C'])
                assert row['id'] == original['id'] and row['accuracy'] == pytest.approx(original['accuracy'])
        verify_feedback(interface, result['feedback'], inputs)
        if interface == 'compiled':
            assert set(result['families']) == {'direction_sizing', 'dlinear_regressor', 'gaussian_hmm_binary', 'lstm_binary', 'random_binary', 'rbf_svm_binary', 'residual_regressor', 'ridge_regressor', 'xgboost_regressor', 'cnn1d_binary', 'entropy_var_regressor', 'feature_link_binary', 'pseudo_label_binary'}
            raw = pl.read_parquet(inputs / 'spot_15m_20250101_20250531.parquet')
            labels = raw.with_columns(((pl.col('close').shift(-1)/pl.col('close')-1)*100).alias('next_return'))
            for family, observed in result['families'].items():
                expected_rounds = 2 if family == 'direction_sizing' else 1
                assert len(observed['records']) == len(observed['metrics']) == expected_rounds
                assert observed['reconstructed_sensors'] == (0 if family == 'random_binary' else 1)
                by_id = {record['round_id']: record for record in observed['records']}
                assert len(by_id) == expected_rounds
                for metric in observed['metrics']:
                    record = by_id[metric['id']]
                    predictions = np.asarray(observed['predictions'][record['round_id']], dtype=float)
                    assert predictions.size == record['preds_ref']['count']
                    assert metric['id'] == record['round_id'] and predictions.size > 300 and np.isfinite(predictions).all()
                    if family.endswith('_regressor'):
                        alignment = record['alignment']
                        horizon_labels = raw.with_columns(((pl.col('close').shift(-96) / pl.col('close') - 1) * 100).alias('next_return')) if family == 'entropy_var_regressor' else labels
                        rows = horizon_labels.filter(pl.col('datetime').is_between(datetime.fromisoformat(alignment['first_test_datetime']), datetime.fromisoformat(alignment['last_test_datetime'])))
                        assert rows.height == predictions.size
                        expected_rmse = float(np.sqrt(np.mean((predictions-rows['next_return'].to_numpy())**2)))
                        assert metric['rmse'] == pytest.approx(expected_rmse, rel=1e-10, abs=1e-10)
                        if family == 'entropy_var_regressor':
                            residual = rows['next_return'].to_numpy() - predictions
                            assert metric['pinball_loss'] == pytest.approx(np.mean(np.maximum(.05 * residual, -.95 * residual)))
                            assert metric['violation_rate'] == pytest.approx(np.mean(residual < 0))
                    elif family == 'random_binary':
                        assert set(predictions) == {0, 1} and 0 <= metric['accuracy'] <= 1
                    elif family in ('rbf_svm_binary', 'gaussian_hmm_binary', 'lstm_binary', 'cnn1d_binary', 'feature_link_binary', 'pseudo_label_binary'):
                        alignment = record['alignment']
                        rows = labels.filter(pl.col('datetime').is_between(datetime.fromisoformat(alignment['first_test_datetime']), datetime.fromisoformat(alignment['last_test_datetime'])))
                        assert rows.height == predictions.size and set(predictions) <= {0, 1}
                        assert metric['accuracy'] == pytest.approx(np.mean(predictions == (rows['next_return'].to_numpy() > 0)))
                    else:
                        assert np.abs(predictions).max() <= 1 and 0 <= metric['direction_accuracy'] <= 1
                        assert len(record['trade_contract_digest']) == 64 and record['learning_binding']
    elif name == 'recovery':
        assert result['interrupted_rounds'] == 2 and result['rounds'] == 4
        assert result['round_indices'] == [0, 1, 2, 3]
        assert len(set(result['round_ids'])) == 4
        assert (output / 'experiment/checkpoint.json').is_file()
        assert result['uninterrupted_equivalence'] is True
        resumed = pl.read_csv(output / 'experiment/results.csv').drop('execution_time')
        uninterrupted = pl.read_csv(output / 'uninterrupted/results.csv').drop('execution_time')
        assert resumed.equals(uninterrupted)
        def records(directory):
            with gzip.open(directory / 'round_data.jsonl.gz', 'rt') as stream:
                return [json.loads(line) for line in stream]
        assert records(output / 'experiment') == records(output / 'uninterrupted')
        if interface == 'python':
            assert result['fitted_state_restored'] is False
        else:
            assert result['reconstructed_sensors'] == 1
    elif name == 'walk_forward':
        if interface == 'python':
            bars = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=384)
            assert len(result['folds']) == 4
            assert [(f['anchored'], f['fold'], f['train_rows']) for f in result['folds']] == [(True, 0, 284), (True, 1, 332), (False, 0, 284), (False, 1, 284)]
            for fold in result['folds']:
                assert fold['test_rows'] == len(fold['predictions']) == 48
                assert datetime.fromisoformat(fold['train_end']) < datetime.fromisoformat(fold['test_start'])
                assert all(math.isfinite(value) for value in fold['predictions'])
                candidates = [(fold_ridge(bars, fold, alpha)[0], alpha) for alpha in (.1, 1.)]
                score, alpha = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
                assert fold['alpha'] == alpha and fold['validation_net_return'] == pytest.approx(score, rel=1e-10, abs=1e-12)
                np.testing.assert_allclose(fold['predictions'], fold_ridge(bars, fold, alpha)[1], rtol=1e-10, atol=1e-12)
                test = bars[288+fold['fold']*48:336+fold['fold']*48]
                assert fold['test_open'] == test['open'].to_list() and fold['test_close'] == test['close'].to_list()
                assert fold['available_at_ns'] == (test['datetime'].dt.epoch('ns')+3_600_000_000_000).to_list()
        else:
            verify_acceptance(result['acceptance'], output / 'acceptance', inputs)
            assert result['rounds'] == 1 and len(result['folds']) == 2
            assert result['member_folds'] == [0, 1]
            assert len(result['member_ids']) == 2 and all(len(item) == 4 for item in result['member_ids'])
            assert len(set(result['selection_cutoffs'])) == 1
            assert all(datetime.fromisoformat(fitting) < datetime.fromisoformat(result['selection_cutoffs'][0]) for fitting in result['fitting_cutoffs'])
            assert result['objective'] == pytest.approx(np.mean([row['val_backtest_total_return'] for row in result['folds']]))
            assert all(row['val_score'] == row['val_backtest_total_return'] for row in result['folds'])
            record_file = output / result['round_records']['path']
            assert hashlib.sha256(record_file.read_bytes()).hexdigest() == result['round_records']['sha256']
            with gzip.open(record_file, 'rt') as stream:
                fold_records = [json.loads(line) for line in stream]
            assert len(fold_records) == 1 and len(fold_records[0]['folds']) == 2
            for recorded, observed in zip(fold_records[0]['folds'], result['folds'], strict=True):
                for metric in ('backtest_total_return', 'val_backtest_total_return', 'val_score', 'optimal_threshold'):
                    assert np.isfinite(observed[metric])
                    assert recorded['results'][metric] == observed[metric]
    elif name == 'holdout':
        if interface == 'python':
            rows = pl.read_parquet(output / 'assessment.parquet')
            assert result['assessment_rows'] == rows.height == 479
            raw = pl.read_parquet(inputs / 'spot_15m_20250101_20250531.parquet', n_rows=1920)
            expected = rows['position'].to_numpy()*(raw['close'].to_numpy()[1441:1920]/raw['open'].to_numpy()[1441:1920]-1)-abs(rows['position'].to_numpy())*.0002
            np.testing.assert_allclose(rows['net_return'], expected, rtol=1e-12, atol=1e-12)
            assert result['mean_net_return'] == pytest.approx(float(expected.mean()), rel=1e-12, abs=1e-12)
        else:
            assert result['rounds'] == 1 and result['exposure']['class'] == 'unknown'
            assert result['exposure']['unknown_external_exposure'] is True
            report = json.loads((output / 'retained-report.json').read_text())
            rows = pl.read_parquet(output / 'project' / report['rows_path'])
            final = rows.filter(pl.col('role') == 'final')
            assert final.height == 24
            assert report['classification']['accuracy']['denominator'] == 23
            assert len(report['scenarios']) == 1
            assert report['rows_sha256'] == hashlib.sha256((output / 'project' / report['rows_path']).read_bytes()).hexdigest()
    elif name == 'calibration':
        assert len(result['modes']) == 4 and result['validation_reused'] is True
        if interface == 'python':
            verify_composition(result['composition'])
            assert {row['method'] for row in result['probability_methods']} == {'sigmoid', 'isotonic'}
            original = np.asarray(result['modes'][0]['validation_probabilities'])
            isotonic = next(row for row in result['probability_methods'] if row['method'] == 'isotonic')
            assert (np.diff(np.asarray(isotonic['validation_probabilities'])[np.argsort(original)]) >= -1e-14).all()
            for row in [*result['modes'], *result['probability_methods']]:
                assert row['inference_rows'] == 480 and 0 <= row['positive_predictions'] <= 480
                probabilities = np.asarray(row['validation_probabilities'])
                truth = np.asarray(row['validation_labels'])
                assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
                if row['tuned']:
                    choices = []
                    for threshold in np.minimum(np.arange(.2, .7+.05, .05), .7):
                        predictions = probabilities >= threshold
                        if predictions.sum():
                            precision = np.sum(predictions & (truth == 1)) / predictions.sum()
                            choices.append((float(precision*np.sqrt(predictions.mean())), float(threshold)))
                    best = max(choices, key=lambda item: item[0])
                    assert row['threshold'] == pytest.approx(best[1]) and row['validation_score'] == pytest.approx(best[0])
                else:
                    assert row['threshold'] == .5 and row['validation_score'] is None
        else:
            assert result['reconstructed_sensors'] == 4
            assert {(row['use_calibration'], row['use_threshold']) for row in result['modes']} == {(a, b) for a in (False, True) for b in (False, True)}
            for row in result['modes']:
                assert (row['val_score'] is None) == (not row['use_threshold'])
    elif name == 'execution':
        hourly = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=168)
        times = hourly['datetime'].dt.epoch('ns')
        bounds = [times[0], times[-1]+3_600_000_000_000] if interface == 'python' else [times[96], times[120]]
        if interface == 'python':
            metrics = result['metrics']
            assert len(result['contract_digest']) == 64 and result['fills']
            assert metrics['total_return'] == pytest.approx(metrics['ending_equity']/10000-1)
            assert metrics['net_pnl'] == pytest.approx(metrics['gross_pnl']-metrics['fees']-metrics['slippage']+metrics['funding_pnl'])
            fills = pl.read_parquet(output / 'fills.parquet')
            assert fills.to_dicts() == result['fills'], 'Native fill records differ from the checked output'
            assert fills.head(8).to_dicts() == result['ledger_excerpt']
            assert metrics['fees'] > 0
            assert result['context']['contract']['policy']['execution_lag_seconds'] == 3600
            assert hashlib.sha256(json.dumps(result['context']['contract'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == result['contract_digest']
            execution_clocks(result['context'], {'fills': result['fills'], 'intents': pl.read_parquet(output / 'intents.parquet').to_dicts()}, hourly)
        else:
            assert result['rounds'] == 2 and len(set(result['contracts'])) == 2
            assert len(result['ledgers']) == 2
            assert {row['risk_span'] for row in result['metrics']} == {6, 12}
            assert all(row['backtest_completed_trades'] > 0 for row in result['metrics'])
            with gzip.open(output / 'experiment/round_data.jsonl.gz', 'rt') as stream:
                native = [json.loads(line) for line in stream]
            assert result['ledgers'] == [row['trade_ledger'] for row in native]
            for context, ledger in zip(result['contexts'], result['ledgers'], strict=True):
                execution_clocks(context, ledger, hourly)
        funded = result['funded']
        raw = json.loads((inputs / 'btcusdt_funding_20240101_20240108.json').read_text())['records']
        assert funded['recorded_events'] == len(raw) == 22
        assert funded['input_sha256'] == hashlib.sha256((inputs / 'btcusdt_funding_20240101_20240108.json').read_bytes()).hexdigest()
        observed_rates = {row['fundingTime'] * 1_000_000: (float(row['fundingRate']), float(row['markPrice'])) for row in raw}
        ledgers = [{'funding': funded['payments'], 'fills': funded['fills'], 'intents': pl.read_parquet(output / 'funded-intents.parquet').to_dicts()}] if interface == 'python' else funded['ledgers']
        contexts = [funded['context']] if interface == 'python' else funded['contexts']
        if interface == 'python':
            assert pl.read_parquet(output / 'funded-fills.parquet').to_dicts() == funded['fills'], 'Native funded fill records differ from the checked output'
            assert funded['context']['contract']['policy']['execution_lag_seconds'] == 3600
            assert hashlib.sha256(json.dumps(funded['context']['contract'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == funded['contract_digest']
        else:
            with gzip.open(output / 'funded/round_data.jsonl.gz', 'rt') as stream:
                native = [json.loads(line) for line in stream]
            assert funded['ledgers'] == [row['trade_ledger'] for row in native]
        for context, ledger in zip(contexts, ledgers, strict=True):
            assert context['partition'] == bounds
            execution_clocks(context, ledger, hourly)
            assert ledger['funding']
            expected = sorted(time for time in observed_rates if bounds[0] <= time <= bounds[1])
            assert [row['end_ns'] for row in ledger['funding']] == expected
            assert any(payment['recognized_delta'] != 0 for payment in ledger['funding'])
            for payment in ledger['funding']:
                rate, mark = observed_rates[payment['end_ns']]
                quantity = sum(fill['quantity_delta'] for fill in ledger['fills'] if fill['time_ns'] < payment['end_ns'])
                assert payment['quantity'] == pytest.approx(quantity, rel=1e-12, abs=1e-12)
                assert payment['rate_decimal'] == rate and payment['valuation_price'] == mark
                assert payment['recognized_delta'] == pytest.approx(-payment['quantity'] * rate * mark, rel=1e-12, abs=1e-12)
        if interface == 'python':
            assert funded['metrics']['funding_pnl'] == pytest.approx(sum(row['recognized_delta'] for row in funded['payments']))
        else:
            assert len(funded['metrics']) == len(funded['ledgers']) == 2
            for row, ledger in zip(funded['metrics'], funded['ledgers']):
                assert row['backtest_funding_pnl'] == pytest.approx(sum(payment['recognized_delta'] for payment in ledger['funding']))
    elif name == 'bars':
        if interface == 'python':
            raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=384)
            assert {row['kind'] for row in result['formed']} == {'volume', 'trade', 'liquidity'}
            for row in result['formed']:
                retained = raw.head(row['source_rows_in_complete_bars'])
                assert 0 < row['rows'] < row['source_rows_in_complete_bars'] <= 384
                assert row['volume'] == pytest.approx(retained['volume'].sum(), rel=1e-12)
                assert row['first_open'] == retained['open'][0] and row['last_close'] == retained['close'][-1]
                actual = pl.read_parquet(output / (row['kind'] + '.parquet'))
                column = {'volume': 'volume', 'trade': 'no_of_trades', 'liquidity': 'liquidity_sum'}[row['kind']]
                cursor = 0
                for bar in actual.iter_rows(named=True):
                    part = raw.slice(cursor, bar['bar_count'])
                    assert sum(part[column]) >= row['threshold'] and sum(part[column].to_list()[:-1]) < row['threshold']
                    assert bar['datetime'] == part['datetime'][0] and bar['base_interval'] == 3600
                    assert bar['open'] == part['open'][0] and bar['close'] == part['close'][-1]
                    assert bar['high'] == part['high'].max() and bar['low'] == part['low'].min()
                    for field in ('volume', 'no_of_trades', 'liquidity_sum'):
                        assert bar[field] == pytest.approx(part[field].sum(), rel=1e-12)
                    cursor += bar['bar_count']
                assert cursor == row['source_rows_in_complete_bars'] and sum(raw.slice(cursor)[column]) < row['threshold']
                assert row['first_start'] == actual['datetime'][0].isoformat()
                assert row['last_start'] == actual['datetime'][-1].isoformat()
                assert row['last_available'] == (raw['datetime'][cursor - 1] + timedelta(hours=1)).isoformat()
        else:
            assert result['rounds'] == 3 and {row['bar_type'] for row in result['modes']} == {'volume', 'trade', 'liquidity'}
        profile = result['trade_profile']
        from limen import HistoricalData
        historical = HistoricalData()
        historical.get_any_file(str(inputs / 'binance_trades_btcusdt_2025-05-23_90s.zip'), has_header=False, columns=['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null'])
        trades = historical.data
        assert isinstance(trades, pl.DataFrame) and profile['trade_rows'] == trades.height
        assert len(profile['bars']) == 3
        native = pl.read_parquet(output / 'trade-profile.parquet')
        buckets = sorted(set((trades['datetime'].dt.epoch('ns')//30_000_000_000*30_000_000_000).to_list()))
        assert native['datetime'].dt.epoch('ns').to_list() == buckets
        count = 0
        for start, row, retained in zip(buckets, native.to_dicts(), profile['bars'], strict=True):
            part = trades.filter(pl.col('datetime').dt.epoch('ns').is_between(start, start+30_000_000_000, closed='left'))
            count += part.height
            assert part.height > 0 and row['datetime'] == datetime.fromisoformat(retained['datetime'])
            assert row['open'] == part['price'][0] and row['close'] == part['price'][-1]
            assert row['high'] == part['price'].max() and row['low'] == part['price'].min()
            assert row['volume'] == pytest.approx(part['quantity'].sum(), rel=1e-12)
            assert row['value_sum'] == pytest.approx((part['price']*part['quantity']).sum(), rel=1e-12)
        assert count == profile['trade_rows']
        assert sum(row['volume'] for row in profile['bars']) == pytest.approx(trades['quantity'].sum())
        assert sum(row['value_sum'] for row in profile['bars']) == pytest.approx((trades['price']*trades['quantity']).sum())
        for row in profile['bars']:
            assert row['vwap'] * row['volume'] == pytest.approx(row['value_sum'])
            assert all(math.isfinite(row[column]) and row[column] >= 0 for column in ('flux_rel_std_mean', 'flux_rel_std_var', 'entropy_mean', 'entropy_var', 'Δflux_rms', 'Δentropy_rms'))
    elif name == 'observed_costs':
        assert interface == 'python'
        calibration = result['calibration']
        assert calibration['support']['selected_legs'] == calibration['support']['admitted_legs'] == 363
        assert calibration['support']['low_effective_support_sides'] == ['sell']
        assert calibration['clocks']['historical_delivery'] == 'unknown'
        assert {(row['mode'], row['hour']) for row in result['runs']} == {(mode, hour) for mode in ('binary', 'target_exposure') for hour in (1, 2)}
        benchmark = json.loads((inputs / 'observed_costs/benchmark.json').read_text())
        digests = []
        for row in result['runs']:
            comparison = row['comparison']
            assert row['rounds'] == 1 and comparison['calibration_digest'] == calibration['digest']
            candidate = comparison['candidates']['caller']
            digests.append((candidate['input_digest'], candidate['application_digest']))
            original, calibrated = [candidate['scenarios'][name] for name in ('original', 'calibrated')]
            assert calibrated['disposition'] == 'retrospective_low_effective_support'
            assert 'time_extrapolation' in calibrated['support']['flags']
            policies = [scenario['contract']['trade_contract']['policy'].copy() for scenario in (original, calibrated)]
            assert policies[0].pop('slip_bps') == 5.0
            assert policies[1].pop('slip_bps') == pytest.approx(float(calibration['estimates']['slip_bps']))
            assert policies[0] == policies[1] and policies[0]['fee_bps'] == 5.0
            assert policies[0]['prediction_mode'] == row['mode']
            for scenario in (original, calibrated):
                ledger = scenario['ledger']
                assert ledger['fills'] and scenario['metrics']['completed_trades'] > 0
                for fill in ledger['fills']:
                    recorded = next(point for point in benchmark if point['row_id'] == fill['evidence_row_id'])
                    assert fill['reference_price'] == recorded['price'] and fill['time_ns'] == recorded['time_ns']
                    assert fill['fee'] == pytest.approx(abs(fill['quantity_delta'] * fill['fill_price']) * .0005)
            saved = json.loads((output / f"{row['mode']}-{row['hour']}/comparison.json").read_text())
            assert saved['digest'] == comparison['digest']
        assert len(set(digests)) == 4
    elif name == 'allocation':
        assert interface == 'compiled' and result['rounds'] == 2
        assert all(row['allocation_training_completed'] >= 2 for row in result['results'])
        assert set(result['cases']) == {'rule', 'ml', 'funded', 'unfunded'}
        raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet')
        for name, case in result['cases'].items():
            assert case['primary'] == case['plain_primary'] and case['exposure'] == case['repeated']
            assert len(case['exposure']) == len(case['primary']) == len(case['row_ids'])
            assert case['binding']['batch_digests']['test'] == case['batch_digest']
            assert case['binding']['primary']['reference'].endswith('.logreg_binary' if name == 'ml' else '.rule_based')
            policy = case['contract']['policy']
            parameters = case['binding']['allocation_params']
            held, previous = 0., 0.
            for decision, exposure, probability, gain, stop in zip(case['primary'], case['exposure'], case['take_probability'], case['signed_return'], case['stops'], strict=True):
                if decision == 0:
                    held = 0.
                elif previous == 0:
                    risk = parameters['risk_budget_fraction'] / ((stop + 2*policy['fee_bps'] + 2*policy['slip_bps']) / 10000)
                    held = min(parameters['max_size'], policy['max_exposure'], max(0., gain/parameters.get('return_scale', .01)), risk) if probability >= parameters['probability_threshold'] else 0.
                    held = 0. if held <= policy['flat_threshold'] else held
                assert exposure == pytest.approx(held, abs=1e-12)
                previous = decision
            training = pl.read_parquet(output / name / 'training.parquet')
            complete = training.filter(pl.col('available'))
            assert complete.height == case['completed'] >= 2 and training['opportunity_id'].n_unique() == training.height
            np.testing.assert_allclose(complete['net_return_on_entry_notional'], complete['net_pnl']/complete['entry_notional'])
            binding = case['binding']['training']
            for fold in binding['folds']:
                rows = complete.filter(pl.col('opening_block') == fold['block'])
                assert rows['label_available_ns'].lt(fold['end_ns']).all()
                assert rows['decision_available_at_ns'].ge(fold['start_ns']).all()
                if name == 'ml':
                    assert fold['fit_prefix_rows'] < case['training_rows']
            window = case['training_window']
            start, end = [datetime.fromisoformat(window[key]).replace(tzinfo=timezone.utc) for key in ('train_start', 'train_end')]
            source = raw.filter(pl.col('datetime').is_between(start, end, closed='left'))
            assert source.height == case['training_rows']
            checksum = hashlib.sha256(str(source.schema).encode() + source.hash_rows(seed=0).to_numpy().tobytes()).hexdigest()
            assert checksum == binding['raw_train']
            if name == 'ml':
                probe = case['prefix']
                assert probe['prefix_rows'] == binding['folds'][0]['fit_prefix_rows']
                digest = hashlib.sha256(json.dumps({'predictions': probe['predictions']}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                assert digest == binding['folds'][0]['primary_digest']
        funded, unfunded = [pl.read_parquet(output / name / 'training.parquet').filter(pl.col('available')) for name in ('funded', 'unfunded')]
        assert funded['opportunity_id'].to_list() == unfunded['opportunity_id'].to_list()
        assert funded['net_pnl'].ne(unfunded['net_pnl']).any()
        payments = pl.read_parquet(output / 'funded/funding.parquet')
        assert payments.height > 0
        np.testing.assert_allclose(payments['cash_delta'], -payments['quantity']*payments['rate_decimal']*payments['valuation_price'])
    elif name == 'command_line':
        assert interface == 'compiled'
        commands = [item['args'][0] for item in result['commands']]
        assert set(commands) == {'--version', 'list-templates', 'init', 'validate', 'profile', 'run', 'commit', 'ls', 'lineage', 'fork', 'reindex', 'backup', 'new'}
        assert commands.count('run') == 3 and commands.count('commit') == 2
        assert all(item['returncode'] == 0 for item in result['commands'])
        project = output / 'project'
        index = json.loads((project / 'manifests/committed/index.json').read_text())
        assert result['manifests'] == index['manifests'] and len(index['manifests']) == 2
        child = next(item for item in index['manifests'] if item['id'] != result['parent_id'])
        assert child['parent_id'] == result['parent_id']
        directory = output / result['run']
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert metadata['manifest_id'] == result['parent_id']
        csv, parquet = pl.read_csv(directory / 'results.csv'), pl.read_parquet(directory / 'results.parquet')
        assert csv.columns == parquet.columns and csv.height == parquet.height == 4
        assert set(csv['C']) == {.1, .5, 1., 2.} and csv['id'].n_unique() == 4
        for column in csv.columns:
            if csv[column].dtype.is_numeric():
                np.testing.assert_allclose(csv[column].to_numpy(), parquet[column].to_numpy(), equal_nan=True)
            else:
                assert csv[column].to_list() == parquet[column].to_list()
        restored = output / 'restored'
        original_names = set(subprocess.check_output(['git', '-C', str(project), 'ls-files', '-z'], text=True).split('\0')) - {''}
        restored_names = set(subprocess.check_output(['git', '-C', str(restored), 'ls-files', '-z'], text=True).split('\0')) - {''}
        assert set(result['project_files']) == original_names == restored_names
        assert 'manifests/holdout-lease.sqlite3' not in original_names
        for name, expected in result['project_files'].items():
            assert hashlib.sha256((project / name).read_bytes()).hexdigest() == expected
            assert hashlib.sha256((restored / name).read_bytes()).hexdigest() == expected
    else:
        raise AssertionError('Unproved scenario: ' + name)


def verify_preparation(interface, result, output, inputs):
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
    assert result['partition_rows'] == [864, 288, 288]
    assert result['carried_context'] is (interface == 'compiled')
    assert result['fixed_d'] == .4 and result['weight_threshold'] == .05
    assert [case['name'] for case in result['cases']] == ['baseline', 'compressed']
    if interface == 'python':
        assert result['adf_grid_result'] in (0., .2, .4)
        assert 0 <= result['adf_p_value'] <= 1
        assert result['adf_stationary'] is (result['adf_p_value'] <= .05)
    weights = [1.]
    while True:
        candidate = -weights[-1] * (.4-len(weights)+1) / len(weights)
        if abs(candidate) < .05:
            break
        weights.append(candidate)
    for case in result['cases']:
        compressed = case['name'] == 'compressed'
        columns = ['close', 'roc_3', 'close_fracdiff'] if compressed else ['close', 'roc_3', 'roc_12', 'close_fracdiff']
        assert case['features'] == columns and case['dropped'] == (['roc_12'] if compressed else [])
        frames = [pl.read_parquet(output / case['name'] / (split + '.parquet')) for split in ('train', 'val', 'test')]
        assert [frame.height for frame in frames] == case['rows']
        assert all(frame.height > 200 for frame in frames)
        expected_surfaces = []
        fit_population = None
        for split, (start, end), frame in zip(('train', 'val', 'test'), ((0, 864), (864, 1152), (1152, 1440)), frames, strict=True):
            context = (8 if compressed else 12) if interface == 'compiled' and split != 'train' else 0
            source = raw[start-context:end]
            close = source['close'].to_numpy()
            clock = source['datetime'].dt.epoch('ns').to_list()
            indices = [clock.index(value) for value in frame['datetime'].dt.epoch('ns').to_list()]
            assert indices == sorted(set(indices))
            assert all(start <= start-context+index < end for index in indices)
            feature_values = {'close': close, 'roc_3': np.r_[np.full(3, np.nan), (close[3:]/close[:-3]-1)*100],
                              'roc_12': np.r_[np.full(12, np.nan), (close[12:]/close[:-12]-1)*100]}
            feature_values['close_fracdiff'] = np.r_[np.full(len(weights)-1, np.nan), np.convolve(close, weights, mode='valid')]
            unscaled = np.column_stack([feature_values[column][indices] for column in columns])
            assert np.isfinite(unscaled).all()
            scaled = (unscaled-np.array([case['medians'][column] for column in columns])) / np.array([case['iqrs'][column] for column in columns])
            expected_surfaces.append(scaled)
            expected_matrix = (scaled-np.asarray(case['pca_mean'])) @ np.asarray(case['components']).T if compressed else scaled
            actual_columns = ['pc_0', 'pc_1'] if compressed else columns
            assert frame.columns == [*actual_columns, 'datetime', 'label'] or frame.columns == ['datetime', *actual_columns, 'label']
            np.testing.assert_allclose(frame.select(actual_columns).to_numpy(), expected_matrix, rtol=1e-10, atol=1e-9)
            returns = close[1:]/close[:-1]-1
            labels = []
            for index in indices:
                assert index >= 8
                decay = (1-2/9)**np.arange(index-1, -1, -1)
                mean = np.average(returns[:index], weights=decay)
                variance = np.average((returns[:index]-mean)**2, weights=decay) / (1-np.sum(decay**2)/np.sum(decay)**2)
                sigma = math.sqrt(variance)
                label = None
                for ahead in range(1, min(3, len(close)-1-index)+1):
                    move = close[index+ahead]/close[index]-1
                    if move >= 1.5*sigma:
                        label = 1
                        break
                    if move <= -sigma:
                        label = -1
                        break
                if label is None and index+3 < len(close):
                    label = 0
                labels.append(label)
            assert labels == frame['label'].to_list(), 'Labels differ from the independently scanned forward-close geometry'
            if split == 'train':
                fit_indices = range(indices[0], len(close)) if interface == 'compiled' else indices
                fit_population = np.column_stack([feature_values[column][list(fit_indices)] for column in columns])
        assert fit_population is not None
        for index, column in enumerate(columns):
            values = fit_population[:, index]
            assert case['medians'][column] == pytest.approx(float(np.median(values)), rel=1e-12)
            lo, hi = np.quantile(values, [.25, .75], method='nearest')
            assert case['iqrs'][column] == pytest.approx(float(hi-lo) if hi != lo else 1., rel=1e-12)
        if compressed:
            components = np.asarray(case['components'])
            np.testing.assert_allclose(case['pca_mean'], expected_surfaces[0].mean(axis=0), rtol=1e-10, atol=1e-10)
            _, _, vectors = np.linalg.svd(expected_surfaces[0]-expected_surfaces[0].mean(axis=0), full_matrices=False)
            np.testing.assert_allclose(components.T@components, vectors[:2].T@vectors[:2], rtol=1e-9, atol=1e-9)
        x_train = frames[0].drop('datetime', 'label').to_numpy()
        x_test = frames[2].drop('datetime', 'label').to_numpy()
        y = frames[0]['label'].to_numpy()
        centered = x_train-x_train.mean(axis=0)
        coefficients = np.linalg.solve(centered.T@centered+np.eye(centered.shape[1]), centered.T@(y-y.mean()))
        predicted = (x_test-x_train.mean(axis=0))@coefficients+y.mean()
        np.testing.assert_allclose(case['predictions'], predicted, rtol=1e-9, atol=1e-10)
        assert case['rmse'] == pytest.approx(float(np.sqrt(np.mean((predicted-frames[2]['label'].to_numpy())**2))), rel=1e-10)


def verify_common_baselines(interface, result, output, inputs):
    from limen.sfd.rule_based.common_strategies import strategy_signals
    names = {'buy_and_hold', 'sma_cross', 'ema_cross', 'macd', 'rsi_mean_reversion',
             'bollinger_mean_reversion', 'donchian_breakout', 'time_series_momentum'}
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet')
    bars = raw[raw.height*8//11 + raw.height//11:]
    if 'test_start' in result:
        bars = raw.filter(pl.col('datetime').is_between(datetime.fromisoformat(result['test_start']), datetime.fromisoformat(result['test_end']), closed='left'))
        assert bars.height == result['test_rows']
    else:
        assert bars.height == 1598
    assert len(result['rows']) == len(names) == 8
    assert {row['strategy'] for row in result['rows']} == names
    close, opening = bars['close'].to_numpy(), bars['open'].to_numpy()
    for row in result['rows']:
        name = row['strategy']
        signals = strategy_signals(bars, name)
        assert len(signals) == bars.height and signals.dtype == np.int8 and set(signals) <= {0, 1}
        for stop in (201, 401, 801, bars.height-1):
            np.testing.assert_array_equal(strategy_signals(bars[:stop], name), signals[:stop])
        if name == 'buy_and_hold':
            np.testing.assert_array_equal(signals, np.ones(bars.height))
        elif name == 'sma_cross':
            expected = np.zeros(bars.height)
            for i in range(199, bars.height):
                expected[i] = np.mean(close[i-49:i+1]) > np.mean(close[i-199:i+1])
            np.testing.assert_array_equal(signals, expected)
        elif name == 'time_series_momentum':
            np.testing.assert_array_equal(signals, np.r_[np.zeros(20), close[20:] > close[:-20]])
        if interface == 'python':
            assert result['split_rows'] == [6388, 798, 1598] and result['round_trip_bps'] == 2.
            expected = signals[:-1]*(close[1:]/opening[1:]-1-.0002)
            np.testing.assert_allclose(row['net'], expected, rtol=1e-12, atol=1e-12)
            assert row['mean_net_return'] == pytest.approx(float(expected.mean()), rel=1e-12, abs=1e-12)
            retained = pl.read_parquet(output / (name + '.parquet'))
            assert retained['datetime'].equals(bars['datetime'])
            assert retained['desired_position'].to_list() == row['signals'] == signals.tolist()
        else:
            assert result['source_rows'] == 8784 and result['fee_bps'] == 10. and result['slip_bps'] == 5.
            track = result['tracks'][name]
            position = np.r_[0, signals[:-1]].astype(bool)
            expected_gross, expected_net = np.zeros(bars.height), np.zeros(bars.height)
            previous_equity, entry = 1., None
            trades = 0
            for index, held in enumerate(position):
                if not held:
                    entry, previous_equity = None, 1.
                    continue
                if entry is None:
                    entry = close[index-1]
                    trades += 1
                expected_gross[index] = close[index]/close[index-1]-1
                marked = close[index]/(entry*(1+.0005))
                exiting = index == bars.height-1 or not position[index+1]
                equity = marked*(1-.001)*(1-.0005)-.001 if exiting else marked-.001
                expected_net[index] = equity/previous_equity-1
                previous_equity = equity
            np.testing.assert_array_equal(track['pos'], position.astype(float))
            np.testing.assert_allclose(track['gross'], expected_gross, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(track['net'], expected_net, rtol=1e-11, atol=1e-12)
            assert row['num_executed_trades_test'] == trades
            assert row['pnl_per_bar_bps_test'] == pytest.approx(float(expected_net.mean()*10000), abs=1e-9)


def verify_analysis(result, inputs):
    rows = result['rows']
    assert len(rows) == len({(row['window'], row['alpha']) for row in rows}) == 30
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
    close, opening = raw['close'].to_numpy(), raw['open'].to_numpy()
    for row in rows:
        w, alpha = row['window'], row['alpha']
        begin = max(12, w)
        x = close[begin:959]/close[begin-w:959-w]-1
        y = close[begin+1:960]/opening[begin+1:960]-1
        z = (x-x.mean())/x.std()
        coefficient = np.dot(z, y-y.mean())/(np.dot(z,z)+alpha)
        val = close[960+begin:1439]/close[960+begin-w:1439-w]-1
        predicted = (val-x.mean())/x.std()*coefficient+y.mean()
        truth = close[961+begin:1440]/opening[961+begin:1440]-1
        assert row['validation_rmse'] == pytest.approx(float(np.sqrt(np.mean((predicted-truth)**2))), rel=1e-10, abs=1e-12)
    assert {row['feature'] for row in result['correlation']} == {'window', 'alpha'}
    assert {row['parameter'] for row in result['importance']} == {'window', 'alpha'}
    assert all(row['n_configurations'] == 30 and row['n_train']+row['n_evaluation'] == 30 for row in result['importance'])
    for name, control in result['controls'].items():
        assert control['rows'] and control['checkpoint']['yielded_count'] == len(control['rows'])
        assert all(cycle['errors'] == [] for cycle in control['audit'])
        actions = [item for cycle in control['audit'] for item in [*cycle['interventions'], *cycle['suggestions']]]
        assert actions and all(item['source'] == name.title()+'Reducer' for item in actions)
        if name == 'sanity':
            assert all(item.get('action') == 'suggest' and 'execution_timeout' in item['reason'] for item in actions)
        if name == 'focus':
            assert {'set_filter', 'inject_value'} <= {item['op'] for item in actions}
        if name == 'saturation':
            assert any(item['op'] == 'set_filter' for item in actions)


def verify_composition(composition):
    result = composition['results']
    assert composition['retained_state_unchanged'] is True
    assert set(composition['refusals']) == {'duplicate', 'missing_mapping_identity', 'unaligned_members', 'reverse_decisions'}
    assert all(composition['refusals'].values())
    assert result['empty'] == [] and result['last'] == result['mean'][-1]
    assert len(result['members']) == 4
    for index, (average, vote, signed, later) in enumerate(zip(result['mean'], result['vote'], result['signed'], result['later'], strict=True)):
        members = [tape[index] for tape in result['members']]
        assert len({row['datetime'] for row in members}) == 1
        assert average['datetime'] == vote['datetime'] == signed['datetime'] == members[0]['datetime']
        assert average['available_at_ns'] == max(row['available_at_ns'] for row in members)
        if any(row['reason'] is not None for row in members):
            assert all(row['prediction'] is None and row['probability'] is None for row in (average, vote, signed))
        else:
            predictions = np.asarray([row['prediction'] for row in members])
            assert average['prediction'] == pytest.approx(float(predictions.mean()))
            assert average['probability'] == pytest.approx(float(np.mean([row['probability'] for row in members])))
            assert vote['prediction'] == int(predictions.sum() > 2) and vote['probability'] is None
            assert signed['prediction'] == float(np.median(2*predictions-1)) and signed['probability'] is None
        if index <= 12:
            assert later['reason'] == 'inside-training-window' and later['prediction'] is None


def verify_source_formats(result, inputs):
    assert set(result) == {'csv', 'zip', 'arrow'}
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
    for suffix, record in result.items():
        assert record['source_sha256'] == hashlib.sha256((inputs / ('recorded-hourly.'+suffix)).read_bytes()).hexdigest()
        assert record['rows'] == raw.height and record['timestamps_ns'] == raw['datetime'].dt.epoch('ns').to_list()
        assert record['values'] == raw.select('open', 'high', 'low', 'close', 'volume').to_dicts()
