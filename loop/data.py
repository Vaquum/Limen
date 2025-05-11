from typing import Sequence, Dict, List, Tuple
from itertools import accumulate

from vaquum_tools import get_raw_trades_data
from loop.utils.get_klines_historical import get_klines_historical

import pandas as pd
import polars as pl


class HistoricalData:
    
    def __init__(self):

        pass

    def get_historical_klines(self,
                              data_start_date: str,
                              data_end_date: str,
                              data_interval: str) -> None:
        
        '''Get historical klines data from Binance API

        Args:
            data_start_date (str): e.g. '2025-01-01'
            data_end_date (str): e.g. '2025-01-31'
            data_interval (str): Either '1h' or '1d'

        Returns:
            self.data (pl.DataFrame)
    
        '''

        self.data = get_klines_historical(data_interval,
                                            data_start_date,
                                            data_end_date)

        self._int_cols = ['open_time', 'close_time', 'num_trades']

        self._float_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'qav', 'taker_base_vol', 'taker_quote_vol', 'ignore']

        self.data['open_time'] = pd.to_datetime(self.data['open_time'])

        all_cols = self._int_cols + self._float_cols + ['open_time']
        assert set(self.data.columns) == set(all_cols), 'Input data columns do not match the expectation.'
        
        self.data = pl.from_pandas(self.data)

    def get_historical_trades(self,
                              month_year: Tuple = None,
                              n_rows: int = None,
                              include_datetime_col: bool = True) -> None:

        '''Get historical trades data from `tdw.binance_trades`

        Args:
            month_year (Tuple): The month of data to be pulled e.g. (3, 2025)
            n_rows (int): Number of rows to be pulled
            include_datetime_col (bool): If datetime column is to be included

        Returns:
            self.data (pl.DataFrame)
    
        '''
        
        self.data = get_raw_trades_data(month_year=month_year,
                                        n_rows=n_rows,
                                        include_datetime_col=include_datetime_col)

    def split_sequential(self, ratios: Sequence[int]) -> List[pl.DataFrame]:

        '''Split the data into sequential chunks

        Args:
            ratios (Sequence[int]): The ratios of the data to be split

        Returns:
            List[pl.DataFrame]
        '''

        total = self.data.height
        total_ratio = sum(ratios)
        bounds = [int(total * c / total_ratio) for c in accumulate(ratios)]
        starts = [0] + bounds[:-1]
        
        return [self.data.slice(start, end - start) for start, end in zip(starts, bounds)]
    
    def split_random(self, ratios: Sequence[int], seed: int = None) -> List[pl.DataFrame]:

        '''Split the data into random chunks

        Args:
            ratios (Sequence[int]): The ratios of the data to be split
            seed (int): The seed for the random number generator

        Returns:
            List[pl.DataFrame]    
        '''

        total = self.data.height
        total_ratio = sum(ratios)
        bounds = [int(total * c / total_ratio) for c in accumulate(ratios)]
        starts = [0] + bounds[:-1]
        
        return [self.data.sample(fraction=1.0, seed=seed, shuffle=True).slice(start, end - start) for start, end in zip(starts, bounds)]
    