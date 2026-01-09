from limen.features.conserved_flux_renormalization import conserved_flux_renormalization
from limen.data import HistoricalData


def test_conserved_flux_renormalization():

    file_url = 'https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2025-05-23.zip'
    cols = ['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null']

    historical = HistoricalData()
    historical.get_binance_file(file_url, has_header=False, columns=cols)
    trades_df = historical.data

    _ = conserved_flux_renormalization(trades_df)
