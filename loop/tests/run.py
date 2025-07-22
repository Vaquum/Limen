import loop
from loop.models import random, xgboost, lightgbm_example, logreg_example
import uuid
from loop.data import HistoricalData

# Import mega model test
from mega_model_test import test_mega_model_with_live_labeling

# Import confidence filter + create megamodel predictions tests
from test_create_megamodel_predictions import test_create_megamodel_predictions
from test_confidence_filtering_system import (
    test_calibrate_confidence_threshold,
    test_apply_confidence_filtering, 
    test_confidence_filtering_system,
    test_edge_cases
)

from test_quantile_model import test_quantile_model
from test_moving_average_correction_model import test_moving_average_correction
from test_regime_stability import test_regime_stability

from test_account_conviction import test_account_conviction
from test_backtest_conviction import test_backtest_conviction

print("Running Account conviction tests")
test_account_conviction()
print("Running Backtest conviction tests")
test_backtest_conviction()

print(f"Getting historical data")
historical = HistoricalData()
historical.get_historical_klines(n_rows=100000,
                                 kline_size=600,
                                 start_date_limit='2025-04-01',
                                 futures=True)

print(f"Getting historical trades data")
historical.get_historical_trades(n_latest=100)

print("="*80)
print("STARTING ALL TESTS")
print("="*80)

# 1. RUN MEGA MODEL TEST FIRST
print("\nA1. MEGA MODEL TEST (with live labeling)")
print("-" * 50)
mega_results = test_mega_model_with_live_labeling()

print("\nA2. MEGAMODEL PREDICTIONS TEST")
print("-" * 50)
test_create_megamodel_predictions()

print("\nA3. CONFIDENCE FILTERING SYSTEM TEST")
print("-" * 50)
confidence_tests = [
    ("calibrate_confidence_threshold", test_calibrate_confidence_threshold),
    ("apply_confidence_filtering", test_apply_confidence_filtering),
    ("confidence_filtering_system", test_confidence_filtering_system),
    ("edge_cases", test_edge_cases)
]

for test_name, test_func in confidence_tests:
    test_func()

print("\nA4. QUANTILE MODEL TEST")
print("-" * 50)
test_quantile_model()

print("\nA5. MOVING AVERAGE CORRECTION MODEL TEST")
print("-" * 50)
test_moving_average_correction()

print("\nA6. REGIME STABILITY MODEL TEST")
print("-" * 50)
test_regime_stability()

print(f"B1. Running log_df")
print("-" * 50)
from loop.reports.log_df import read_from_file, outcome_df, corr_df
data = read_from_file('logreg_broad_2_3600.csv')
outcome_df = outcome_df(data, ['solver', 'feature_to_drop', 'penalty'], type='categorical')
corr_df = corr_df(outcome_df)

def _get_historical_data(file_url, cols):
    historical = loop.HistoricalData()
    historical.get_binance_file(file_url, False, cols)
    return historical.data

def get_klines_data():
    file_url = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-04.zip'
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'no_of_trades', 'maker_ratio', 'taker_buy_quote_asset_volume', '_null']
    return _get_historical_data(file_url, cols)

def get_trades_data():
    file_url = 'https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2025-05-23.zip'
    cols = ['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null']
    return _get_historical_data(file_url, cols)

def test_binary_metrics():
    data = get_klines_data()
    uel = loop.UniversalExperimentLoop(data=data,
                                      single_file_model=random)
    uel.run(experiment_name=uuid.uuid4().hex[:8],
           n_permutations=2,
           prep_each_round=True)

print(test_binary_metrics())

tests = [(random, get_klines_data, True),
         (xgboost, get_klines_data, False),
         (lightgbm_example, get_trades_data, False),
         (logreg_example, get_klines_data, True)]

for i, test in enumerate(tests, 1):
    print(f"\n  B2.{i} Running {test[0].__name__} with {test[1].__name__}")
    test_name = uuid.uuid4().hex[:8]
    
    try:
        uel = loop.UniversalExperimentLoop(data=test[1](),
                                         single_file_model=test[0])
        uel.run(experiment_name=test_name,
               n_permutations=2,
               prep_each_round=test[2])
        print(f"    ✅ {test[0].__name__}: PASSED")
    except Exception as e:
        print(f"    ❌ {test[0].__name__}: FAILED - {e}")

print("\n" + "="*80)
print("ALL TESTS COMPLETE")
print("="*80)