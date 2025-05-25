from typing import Sequence, List, Tuple, Optional
from itertools import accumulate

from vaquum_tools.binance_file_to_polars import binance_file_to_polars

from .utils.get_klines_data import get_klines_data
from .utils.get_trades_data import get_trades_data
from .utils.get_agg_trades_data import get_agg_trades_data
from .utils.generic_endpoint_for_tdw import generic_endpoint_for_tdw
import polars as pl


class HistoricalData:
    
    def __init__(self):

        pass

    def get_binance_file(self,
                         file_url: str,
                         has_header: bool = False,
                         columns: List[str] = None):

        self.data = binance_file_to_polars(file_url, has_header=has_header)
        self.data.columns = columns

        self.data = self.data.with_columns([
            pl.when(pl.col("timestamp") < 10**13)
            .then(pl.col("timestamp"))
            .otherwise(pl.col("timestamp") // 1000)
            .cast(pl.UInt64) 
            .alias("timestamp")
        ])

        self.data = self.data.with_columns([
            pl.col("timestamp")
            .cast(pl.Datetime("ms"))
            .alias("datetime")
        ])

        self.data_columns = self.data.columns

    def get_historical_klines(self, n_rows: int = None, kline_size: int = 1) -> None:
        
        '''Get historical klines data from Binance API

        Args:
            n_rows (int): Number of rows to be pulled

        Returns:
            self.data (pl.DataFrame)
    
        '''

        self.data = get_klines_data(n_rows=n_rows, kline_size=kline_size)

        self.data_columns = self.data.columns

    def get_historical_trades(self,
                              month_year: Tuple = None,
                              n_latest: int = None,
                              n_random: int = None,
                              include_datetime_col: bool = True) -> None:

        '''Get historical trades data from `tdw.binance_trades`

        Args:
            month_year (Tuple): The month of data to be pulled e.g. (3, 2025)
            n_rows (int): Number of rows to be pulled
            include_datetime_col (bool): If datetime column is to be included

        Returns:
            self.data (pl.DataFrame)
    
        '''
        
        self.data = get_trades_data(month_year=month_year,
                                        n_latest=n_latest,
                                        n_random=n_random,
                                        include_datetime_col=include_datetime_col)
        
        self.data = self.data.with_columns([
            pl.when(pl.col("timestamp") < 10**13)
            .then(pl.col("timestamp"))
            .otherwise(pl.col("timestamp") // 1000)
            .cast(pl.UInt64) 
            .alias("timestamp")
        ])

        self.data_columns = self.data.columns

    def get_historical_agg_trades(self,
                                  month_year: Tuple = None,
                                  n_rows: int = None,
                                  include_datetime_col: bool = True) -> None:

        '''Get historical aggTrades data from `tdw.binance_agg_trades`

        Args:
            month_year (Tuple): The month of data to be pulled e.g. (3, 2025)
            n_rows (int): Number of rows to be pulled
            include_datetime_col (bool): If datetime column is to be included

        Returns:
            self.data (pl.DataFrame)
    
        '''
        
        self.data = get_agg_trades_data(month_year=month_year,
                                        n_rows=n_rows,
                                        include_datetime_col=include_datetime_col)
        
        self.data = self.data.with_columns([
            pl.when(pl.col("timestamp") < 10**13)
            .then(pl.col("timestamp"))
            .otherwise(pl.col("timestamp") // 1000)
            .cast(pl.UInt64) 
            .alias("timestamp")
        ])

        self.data_columns = self.data.columns
        
    def get_historical_futures_trades(self,
                                      month_year: Optional[Tuple[int,int]] = None,
                                      n_rows: Optional[int] = None,
                                      include_datetime_col: bool = True,
                                      show_summary: bool = False) -> pl.DataFrame:
        
        '''Get Binance futures trades data.

        Args:
            month_year (tuple[int,int] | None): (month, year) to fetch, e.g. (3, 2025).
            n_rows (int | None): if set, fetch this many latest rows instead.
            include_datetime_col (bool): whether to include `datetime` in the result.
            show_summary (bool): if a summary for data is printed out

        Returns:
            pl.DataFrame: the requested trades.
        '''

        select_cols = ['futures_trade_id', 'timestamp', 'price', 'quantity', 'is_buyer_maker']
        table_name = 'binance_futures_trades'
        sort_by = 'futures_trade_id'

        return generic_endpoint_for_tdw(month_year=month_year,
                                        n_rows=n_rows,
                                        include_datetime_col=include_datetime_col,
                                        select_cols=select_cols,
                                        table_name=table_name,
                                        sort_by=sort_by,
                                        show_summary=show_summary)

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
    