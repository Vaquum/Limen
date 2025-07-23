"""
Test data utilities for fetching historical data from various sources.
"""

import loop


def _get_historical_data(file_url, cols):
    """
    Helper function to get historical data from a Binance file URL.
    
    Args:
        file_url (str): URL to the Binance data file
        cols (list): List of column names for the data
        
    Returns:
        DataFrame: Historical data
    """
    historical = loop.HistoricalData()
    historical.get_binance_file(file_url, False, cols)
    return historical.data


def get_klines_data():
    """
    Get klines (candlestick) data from Binance.
    
    Returns:
        DataFrame: Klines data with OHLCV and additional columns
    """
    file_url = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-04.zip'
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
            'quote_asset_volume', 'no_of_trades', 'maker_ratio', 'taker_buy_quote_asset_volume', '_null']
    return _get_historical_data(file_url, cols)


def get_trades_data():
    """
    Get trade data from Binance.
    
    Returns:
        DataFrame: Trade data with price, quantity, and timing information
    """
    file_url = 'https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2025-05-23.zip'
    cols = ['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null']
    return _get_historical_data(file_url, cols) 