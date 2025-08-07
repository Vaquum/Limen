'Test data utilities for fetching historical data from various sources.'

import polars as pl
import pandas as pd

import loop


def _get_historical_data(file_url, cols):
    
    '''
    Helper function to get historical data from a Binance file URL.
    
    Args:
        file_url (str): URL to the Binance data file
        cols (list): List of column names for the data
        
    Returns:
        DataFrame: Historical data
    '''
    
    historical = loop.HistoricalData()
    historical.get_binance_file(file_url, False, cols)
    
    return historical.data


def get_klines_data():
    
    '''
    Get klines (candlestick) data from Binance.
    
    Returns:
        DataFrame: Klines data with OHLCV and additional columns
    '''
    df = pd.read_csv('datasets/klines_2h_2020_2025.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = pl.from_pandas(df)
    
    return df


def get_klines_data_small():
    '''
    Get small subset of klines data for faster testing.
    
    Returns:
        DataFrame: Small klines dataset (first 5000 rows)
    '''
    df = pd.read_csv('datasets/klines_2h_2020_2025.csv', nrows=5000)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = pl.from_pandas(df)
    
    return df


def get_trades_data():
    
    '''
    Get trade data from Binance.
    
    Returns:
        DataFrame: Trade data with price, quantity, and timing information
    '''
    
    file_url = 'https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2025-05-23.zip'
    cols = ['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null']
    
    return _get_historical_data(file_url, cols) 