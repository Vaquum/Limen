from datetime import datetime
import hashlib
import json
import math
import subprocess

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


def verify(interface, name, result, output, inputs):
    if name == 'search':
        expected = {'grid', 'random', 'sobol', 'shuffled'} | ({'factorized'} if interface == 'compiled' else set())
        assert set(result['runs']) == expected
        identities = []
        bars = pl.read_parquet(inputs / 'spot_15m_20250101_20250531.parquet', n_rows=1440)
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
        if interface == 'compiled':
            assert set(result['families']) == {'ridge_regressor', 'dlinear_regressor', 'xgboost_regressor', 'random_binary', 'direction_sizing'}
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
                        rows = labels.filter(pl.col('datetime').is_between(datetime.fromisoformat(alignment['first_test_datetime']), datetime.fromisoformat(alignment['last_test_datetime'])))
                        assert rows.height == predictions.size
                        expected_rmse = float(np.sqrt(np.mean((predictions-rows['next_return'].to_numpy())**2)))
                        assert metric['rmse'] == pytest.approx(expected_rmse, rel=1e-10, abs=1e-10)
                    elif family == 'random_binary':
                        assert set(predictions) == {0, 1} and 0 <= metric['accuracy'] <= 1
                    else:
                        assert np.abs(predictions).max() <= 1 and 0 <= metric['direction_accuracy'] <= 1
                        assert len(record['trade_contract_digest']) == 64 and record['learning_binding']
    elif name == 'recovery':
        assert result['interrupted_rounds'] == 2 and result['rounds'] == 4
        assert result['round_indices'] == [0, 1, 2, 3]
        assert len(set(result['round_ids'])) == 4
        assert (output / 'experiment/checkpoint.json').is_file()
        if interface == 'python':
            assert result['fitted_state_restored'] is False
        else:
            assert result['reconstructed_sensors'] == 1
    elif name == 'walk_forward':
        if interface == 'python':
            assert len(result['folds']) == 4
            assert [(f['anchored'], f['fold'], f['train_rows']) for f in result['folds']] == [(True, 0, 284), (True, 1, 332), (False, 0, 284), (False, 1, 284)]
            for fold in result['folds']:
                assert fold['test_rows'] == len(fold['predictions']) == 48
                assert datetime.fromisoformat(fold['train_end']) < datetime.fromisoformat(fold['test_start'])
                assert all(math.isfinite(value) for value in fold['predictions'])
        else:
            assert result['rounds'] == 1 and len(result['folds']) == 2
            assert result['member_folds'] == [0, 1]
            assert len(result['member_ids']) == 2 and all(len(item) == 4 for item in result['member_ids'])
            assert len(set(result['selection_cutoffs'])) == 1
            assert all(datetime.fromisoformat(fitting) < datetime.fromisoformat(result['selection_cutoffs'][0]) for fitting in result['fitting_cutoffs'])
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
            for row in result['modes']:
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
        if interface == 'python':
            metrics = result['metrics']
            assert len(result['contract_digest']) == 64 and result['fills']
            assert metrics['total_return'] == pytest.approx(metrics['ending_equity']/10000-1)
            assert metrics['net_pnl'] == pytest.approx(metrics['gross_pnl']-metrics['fees']-metrics['slippage']+metrics['funding_pnl'])
            fills = pl.read_parquet(output / 'fills.parquet')
            assert fills.height == len(result['fills']) and metrics['fees'] > 0
        else:
            assert result['rounds'] == 2 and len(set(result['contracts'])) == 2
            assert len(result['ledgers']) == 2
            assert {row['risk_span'] for row in result['metrics']} == {6, 12}
            assert all(row['backtest_completed_trades'] > 0 for row in result['metrics'])
        funded = result['funded']
        raw = json.loads((inputs / 'btcusdt_funding_20240101_20240108.json').read_text())['records']
        assert funded['recorded_events'] == len(raw) == 22
        assert funded['input_sha256'] == hashlib.sha256((inputs / 'btcusdt_funding_20240101_20240108.json').read_bytes()).hexdigest()
        observed_rates = {row['fundingTime'] * 1_000_000: (float(row['fundingRate']), float(row['markPrice'])) for row in raw}
        ledgers = [{'funding': funded['payments']}] if interface == 'python' else funded['ledgers']
        for ledger in ledgers:
            assert ledger['funding']
            for payment in ledger['funding']:
                rate, mark = observed_rates[payment['end_ns']]
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
        else:
            assert result['rounds'] == 3 and {row['bar_type'] for row in result['modes']} == {'volume', 'trade', 'liquidity'}
        profile = result['trade_profile']
        from limen import HistoricalData
        historical = HistoricalData()
        historical.get_any_file(str(inputs / 'binance_trades_btcusdt_2025-05-23_90s.zip'), has_header=False, columns=['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null'])
        trades = historical.data
        assert isinstance(trades, pl.DataFrame) and profile['trade_rows'] == trades.height
        assert len(profile['bars']) == 3
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
            source = raw.filter(pl.col('datetime').is_between(datetime.fromisoformat(window['train_start']), datetime.fromisoformat(window['train_end']), closed='left'))
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
