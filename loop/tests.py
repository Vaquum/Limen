import loop
from loop.models import ligthgbm

file_url = 'https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2025-05-23.zip'

historical = loop.HistoricalData()
cols = ['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker',  '_null']
historical.get_binance_file(file_url, False, cols)

uel = loop.UniversalExperimentLoop(historical.data, ligthgbm)
uel.run(experiment_name='test_xx', n_permutations=100)
