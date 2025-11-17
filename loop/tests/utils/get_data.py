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


def get_klines_data_fast():

    '''
    Get optimized klines data for fast testing.
    Uses every 4th row (8h intervals) and limited rows for speed.
    '''

    df = pd.read_csv('datasets/klines_2h_2020_2025.csv', nrows=8000)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.iloc[::4].reset_index(drop=True)  # Every 4th row (8h intervals)
    df = pl.from_pandas(df)
    return df


def get_klines_data_medium():

    '''
    Get medium-sized optimized klines data for models requiring larger datasets.
    Uses full dataset with 6h intervals to ensure sufficient data for regime models.
    '''

    df = pd.read_csv('datasets/klines_2h_2020_2025.csv')  # Use full dataset
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Every 3rd row (6h intervals) -> ~7.8k rows
    df = df.iloc[::3].reset_index(drop=True)
    df = pl.from_pandas(df)
    return df


def get_klines_data_large():

    '''
    Get large dataset for models requiring 20k+ rows (like regime_multiclass).
    Uses full dataset to ensure sufficient data.
    '''

    df = pd.read_csv(
        'datasets/klines_2h_2020_2025.csv')  # Use full dataset (~23k rows)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = pl.from_pandas(df)
    return df


def get_klines_data_small_fast():

    '''
    Get small optimized klines data for fast testing.
    Uses every 6th row (12h intervals) for maximum speed.
    '''
    
    df = pd.read_csv('datasets/klines_2h_2020_2025.csv', nrows=3000)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.iloc[::6].reset_index(drop=True)  # Every 6th row (12h intervals)
    df = pl.from_pandas(df)
    return df
