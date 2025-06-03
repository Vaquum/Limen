import loop
from loop.models import random, xgboost, lightgbm
import uuid


from loop.reports.log_df import read_from_file, outcome_df, corr_df

data = read_from_file('logreg_broad_2_3600.csv')
outcome_df = outcome_df(data, ['solver', 'feature_to_drop', 'penalty'], type='categorical')
corr_df = corr_df(outcome_df)


def _get_klines_data():

    file_url = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-04.zip'
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'no_of_trades', 'maker_ratio', 'taker_buy_quote_asset_volume', '_null']

    historical = loop.HistoricalData()
    historical.get_binance_file(file_url, False, cols)

    return historical.data

def _get_trades_data():

    file_url = 'https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2025-05-23.zip'
    cols = ['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker',  '_null']

    historical = loop.HistoricalData()
    historical.get_binance_file(file_url, False, cols)

    return historical.data

# sfm, data, prep_each_round
tests = [(random, _get_klines_data, True),
         (xgboost, _get_klines_data, False), 
         (lightgbm, _get_trades_data, False)]

for test in tests:

    test_name = uuid.uuid4().hex[:8]
    
    uel = loop.UniversalExperimentLoop(data=test[1](),
                                       single_file_model=test[0])
    
    uel.run(experiment_name=test_name,
            n_permutations=3,
            prep_each_round=test[2])
