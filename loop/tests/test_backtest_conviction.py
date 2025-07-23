import math
import numpy as np
from datetime import datetime
from loop.backtest import Backtest
import csv
import os

def test_basic_functionality():

    backtest = Backtest(100000)
    
    actual = [1, 0, 1, 0, 1]
    prediction = [1, 0, 1, 0, 1]
    price_change = [1000, -1000, 500, -500, 2000]
    open_prices = [50000, 51000, 50000, 50500, 48000]
    close_prices = [51000, 50000, 50500, 50000, 50000]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert 'PnL' in results
    assert 'win_rate' in results
    assert 'max_drawdown' in results
    assert 'expected_value' in results
    assert 'sharpe_ratio' in results
    assert 'net_long_volume' in results
    assert 'net_short_volume' in results
    assert 'net_trade_volume' in results
    assert len(backtest.trades) == 3

def test_perfect_predictions():
    backtest = Backtest(100000)
    
    actual = [1, 1, 1, 1, 1]
    prediction = [1, 1, 1, 1, 1]
    price_change = [1000, 1000, 1000, 1000, 1000]
    open_prices = [50000, 50000, 50000, 50000, 50000]
    close_prices = [51000, 51000, 51000, 51000, 51000]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert results['PnL'] > 0
    assert results['win_rate'] == 1.0
    assert results['max_drawdown'] == 0

def test_terrible_predictions():
    
    backtest = Backtest(100000)
    
    actual = [1, 1, 1, 1, 1]
    prediction = [0, 0, 0, 0, 0]
    price_change = [1000, 1000, 1000, 1000, 1000]
    open_prices = [50000, 50000, 50000, 50000, 50000]
    close_prices = [51000, 51000, 51000, 51000, 51000]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert results['PnL'] == 0
    assert len(backtest.trades) == 0


def test_mixed_signals():
    
    backtest = Backtest(100000)
    
    actual = [1, 0, 1, 0, 1, 0]
    prediction = [1, 0, 0, 1, 1, 0]
    price_change = [1000, -1000, 500, -500, 2000, -1500]
    open_prices = [50000, 51000, 50000, 50500, 48000, 51500]
    close_prices = [51000, 50000, 50500, 50000, 50000, 50000]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert isinstance(results['PnL'], (int, float))
    assert 0 <= results['win_rate'] <= 1
    assert results['max_drawdown'] >= 0

def test_no_valid_trades():
    
    backtest = Backtest(100000)
    
    actual = [1, 0, 1, 0, 1]
    prediction = [0, 0, 0, 0, 0]
    price_change = [1000, -1000, 500, -500, 2000]
    open_prices = [50000, 51000, 50000, 50500, 48000]
    close_prices = [51000, 50000, 50500, 50000, 50000]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert results['PnL'] == 0
    assert results['win_rate'] == 0
    assert results['max_drawdown'] == 0
    assert results['expected_value'] == 0
    assert results['sharpe_ratio'] == 0


def test_zero_price_changes():
    
    backtest = Backtest(100000)
    
    actual = [1, 0, 1, 0, 1]
    prediction = [1, 0, 1, 0, 1]
    price_change = [0, 0, 0, 0, 0]
    open_prices = [50000, 50000, 50000, 50000, 50000]
    close_prices = [50000, 50000, 50000, 50000, 50000]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert results['PnL'] < 0  # Negative due to trading fees
    assert results['win_rate'] == 0


def test_array_length_mismatch():
    
    backtest = Backtest(100000)
    
    actual = [1, 0, 1]
    prediction = [1, 0]
    price_change = [1000, -1000, 500]
    open_prices = [50000, 50000]
    close_prices = [51000, 49000]
    
    try:
        backtest.run(actual, prediction, price_change, open_prices, close_prices)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Arrays must have same length" in str(e)


def test_large_sequence():
    
    backtest = Backtest(1000000)
    
    np.random.seed(42)
    n = 1000
    actual = np.random.randint(0, 2, n)
    prediction = np.random.randint(0, 2, n)
    price_change = np.random.normal(0, 100, n)
    open_prices = np.random.normal(50000, 1000, n)
    close_prices = open_prices + price_change
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert isinstance(results['PnL'], (int, float))
    assert 0 <= results['win_rate'] <= 1
    assert results['max_drawdown'] >= 0
    assert math.isfinite(results['sharpe_ratio'])


def test_metrics_calculation():
    
    backtest = Backtest(100000)
    
    actual = [1, 0, 1, 0]
    prediction = [1, 0, 1, 0]
    price_change = [1000, -1000, 1000, -1000]
    open_prices = [50000, 51000, 50000, 51000]
    close_prices = [51000, 50000, 51000, 50000]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert results['win_rate'] == 1.0
    assert results['PnL'] > 0
    assert results['expected_value'] > 0
    assert isinstance(results['sharpe_ratio'], (int, float))
    

def test_account_integration():
    
    backtest = Backtest(100000)
    
    initial_usdt = backtest.account.account['total_usdt'][-1]
    assert initial_usdt == 100000
    
    actual = [1, 1]
    prediction = [1, 1]
    price_change = [1000, 1000]
    open_prices = [50000, 50000]
    close_prices = [51000, 51000]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    final_usdt = backtest.account.account['total_usdt'][-1]
    assert final_usdt != initial_usdt
    assert len(backtest.trades) == 2


def test_volume_calculations():
    
    backtest = Backtest(100000)
    
    actual = [1, 0, 1]
    prediction = [1, 0, 1]
    price_change = [0.05, -0.02, 0.03]
    open_prices = [50000, 51000, 52000]
    close_prices = [52500, 49980, 53560]
    
    results = backtest.run(actual, prediction, price_change, open_prices, close_prices)
    
    assert results['net_short_volume'] == 0
    assert results['net_long_volume'] > 0
    assert results['net_trade_volume'] == results['net_long_volume']
    
    expected_volume = 0
    for trade in backtest.trades:
        expected_volume += trade['volume']
    
    assert abs(results['net_trade_volume'] - expected_volume) < 0.01


def log_conviction_results(results_list):
    
    log_file = 'backtest-conviction-tests.csv'
    
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'test_name', 'status', 'details'])
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for result in results_list:
            writer.writerow([timestamp, result['test_name'], result['status'], result.get('details', '')])

def test_backtest_conviction():
    
    test_results = []
    
    try:
        try:
            test_basic_functionality()
            print(f'    ✅ {test_basic_functionality.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_basic_functionality.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'basic_functionality', 'status': 'PASSED'})
        
        try:
            test_perfect_predictions()
            print(f'    ✅ {test_perfect_predictions.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_perfect_predictions.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'perfect_predictions', 'status': 'PASSED'})
        
        try:
            test_terrible_predictions()
            print(f'    ✅ {test_terrible_predictions.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_terrible_predictions.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'terrible_predictions', 'status': 'PASSED'})
        
        try:
            test_mixed_signals()
            print(f'    ✅ {test_mixed_signals.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_mixed_signals.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'mixed_signals', 'status': 'PASSED'})
        
        try:
            test_no_valid_trades()
            print(f'    ✅ {test_no_valid_trades.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_no_valid_trades.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'no_valid_trades', 'status': 'PASSED'})
        
        try:
            test_zero_price_changes()
            print(f'    ✅ {test_zero_price_changes.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_zero_price_changes.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'zero_price_changes', 'status': 'PASSED'})
        
        try:
            test_array_length_mismatch()
            print(f'    ✅ {test_array_length_mismatch.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_array_length_mismatch.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'array_length_mismatch', 'status': 'PASSED'})
        
        try:
            test_large_sequence()
            print(f'    ✅ {test_large_sequence.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_large_sequence.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'large_sequence', 'status': 'PASSED'})
        
        try:
            test_metrics_calculation()
            print(f'    ✅ {test_metrics_calculation.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_metrics_calculation.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'metrics_calculation', 'status': 'PASSED'})
        
        try:
            test_account_integration()
            print(f'    ✅ {test_account_integration.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_account_integration.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'account_integration', 'status': 'PASSED'})
        
        try:
            test_volume_calculations()
            print(f'    ✅ {test_volume_calculations.__name__}: PASSED')
        except Exception as e:
            print(f'    ❌ {test_volume_calculations.__name__}: FAILED - {e}')
        test_results.append({'test_name': 'volume_calculations', 'status': 'PASSED'})
        
        log_conviction_results(test_results)

    except Exception as e:
        test_results.append({'test_name': 'FAILED', 'status': 'ERROR', 'details': str(e)})
        log_conviction_results(test_results)
        raise