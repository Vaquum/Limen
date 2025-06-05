import loop
from loop.models import random, xgboost, lightgbm, logreg
import uuid


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
    cols = ['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker',  '_null']

    return _get_historical_data(file_url, cols)

def test_metrics_for_classification():

    data = get_klines_data()
    data = data.head(100000)

    uel = loop.UniversalExperimentLoop(data=data,
                                       single_file_model=random)

    uel.run(experiment_name=uuid.uuid4().hex[:8],
            n_permutations=2,
            prep_each_round=True)

print(test_metrics_for_classification())

# sfm, data, prep_each_round
tests = [(random, get_klines_data, True),
         (xgboost, get_klines_data, False), 
         (lightgbm, get_trades_data, False),
         (logreg, get_klines_data, True)]

for test in tests:

    test_name = uuid.uuid4().hex[:8]
    
    uel = loop.UniversalExperimentLoop(data=test[1](),
                                       single_file_model=test[0])
    
    uel.run(experiment_name=test_name,
            n_permutations=2,
            prep_each_round=test[2])
