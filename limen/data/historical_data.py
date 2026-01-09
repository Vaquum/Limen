import polars as pl

from typing import List, Tuple, Optional

from limen.data._internal.binance_file_to_polars import binance_file_to_polars
from limen.data._internal.generic_endpoints import query_raw_data
from limen.data._internal.generic_endpoints import query_klines_data


class HistoricalData:
    
    def __init__(self, auth_token: str = None):

        '''Set of endpoints to get historical Binance data.'''

        self.auth_token = auth_token

    def get_binance_file(self,
                         file_url: str,
                         has_header: bool = False,
                         columns: List[str] = None):
        
        '''Get historical data from a Binance file based on the file URL. 

        Data can be found here: https://data.binance.vision/

        Args:
            file_url (str): The URL of the Binance file
            has_header (bool): Whether the file has a header
            columns (List[str]): The columns to be included in the data

        Returns:
            self.data (pl.DataFrame)
    
        '''

        self.data = binance_file_to_polars(file_url, has_header=has_header)
        self.data.columns = columns

        self.data = self.data.with_columns([
            pl.when(pl.col('timestamp') < 10**13)
            .then(pl.col('timestamp'))
            .otherwise(pl.col('timestamp') // 1000)
            .cast(pl.UInt64) 
            .alias('timestamp')
        ])

        self.data = self.data.with_columns([
            pl.col('timestamp')
            .cast(pl.Datetime('ms'))
            .alias('datetime')
        ])

        self.data_columns = self.data.columns

    def get_spot_klines(self,
                        n_rows: int = None,
                        kline_size: int = 1,
                        start_date_limit: str = None) -> None:
        
        '''Get historical klines data for Binance spot.

        Args:
            n_rows (int): Number of rows to be pulled
            kline_size (int): Size of the kline in seconds
            start_date_limit (str): The start date of the klines data

        Returns:
            self.data (pl.DataFrame)
    
        '''

        self.data = query_klines_data(n_rows=n_rows,
                                      kline_size=kline_size,
                                      start_date_limit=start_date_limit,
                                      futures=False,
                                      auth_token=self.auth_token)

        self.data_columns = self.data.columns

    def get_futures_klines(self,
                           n_rows: int = None,
                           kline_size: int = 1,
                           start_date_limit: str = None) -> None:
        
        '''Get historical klines data for Binance futures.

        Args:
            n_rows (int): Number of rows to be pulled
            kline_size (int): Size of the kline in seconds
            start_date_limit (str): The start date of the klines data

        Returns:
            self.data (pl.DataFrame)
    
        '''

        self.data = query_klines_data(n_rows=n_rows,
                                      kline_size=kline_size,
                                      start_date_limit=start_date_limit,
                                      futures=True,
                                      auth_token=self.auth_token)

        self.data_columns = self.data.columns

    def get_spot_trades(self,
                        month_year: Tuple = None,
                        n_rows: int = None,
                        n_random: int = None,
                        include_datetime_col: bool = True,
                        show_summary: bool = False) -> None:

        '''Get historical trades data for Binance spot.

        Args:
            month_year (Tuple): The month of data to be pulled e.g. (3, 2025)
            n_rows (int): Number of latest rows to be pulled
            n_random (int): Number of random rows to be pulled
            include_datetime_col (bool): If datetime column is to be included
            show_summary (bool): Print query execution summary

        Returns:
            self.data (pl.DataFrame)

        '''

        self.data = query_raw_data(
            table_name='binance_trades',
            id_col='trade_id',
            select_cols=['trade_id', 'timestamp', 'price', 'quantity', 'is_buyer_maker'],
            month_year=month_year,
            n_rows=n_rows,
            n_random=n_random,
            include_datetime_col=include_datetime_col,
            show_summary=show_summary,
            auth_token=self.auth_token
        )

        self.data_columns = self.data.columns

    def get_spot_agg_trades(self,
                            month_year: Tuple = None,
                            n_rows: int = None,
                            n_random: int = None,
                            include_datetime_col: bool = True,
                            show_summary: bool = False) -> None:

        '''Get historical aggTrades data for Binance spot.

        Args:
            month_year (Tuple): The month of data to be pulled e.g. (3, 2025)
            n_rows (int): Number of latest rows to be pulled
            n_random (int): Number of random rows to be pulled
            include_datetime_col (bool): If datetime column is to be included
            show_summary (bool): Print query execution summary

        Returns:
            self.data (pl.DataFrame)

        '''

        self.data = query_raw_data(
            table_name='binance_agg_trades',
            id_col='agg_trade_id',
            select_cols=[
                'agg_trade_id', 'timestamp', 'price', 'quantity',
                'is_buyer_maker', 'first_trade_id', 'last_trade_id'
            ],
            month_year=month_year,
            n_rows=n_rows,
            n_random=n_random,
            include_datetime_col=include_datetime_col,
            show_summary=show_summary,
            auth_token=self.auth_token
        )

        self.data_columns = self.data.columns

    def get_futures_trades(self,
                           month_year: Optional[Tuple[int,int]] = None,
                           n_rows: Optional[int] = None,
                           n_random: Optional[int] = None,
                           include_datetime_col: bool = True,
                           show_summary: bool = False) -> pl.DataFrame:

        '''Get historical trades data for Binance futures.

        Args:
            month_year (tuple[int,int] | None): (month, year) to fetch, e.g. (3, 2025).
            n_rows (int | None): if set, fetch this many latest rows instead.
            n_random (int | None): if set, fetch this many random rows instead.
            include_datetime_col (bool): whether to include `datetime` in the result.
            show_summary (bool): if a summary for data is printed out

        Returns:
            pl.DataFrame: the requested trades.
        '''

        self.data = query_raw_data(
            table_name='binance_futures_trades',
            id_col='futures_trade_id',
            select_cols=['futures_trade_id', 'timestamp', 'price', 'quantity', 'is_buyer_maker'],
            month_year=month_year,
            n_rows=n_rows,
            n_random=n_random,
            include_datetime_col=include_datetime_col,
            show_summary=show_summary,
            auth_token=self.auth_token
        )

        self.data_columns = self.data.columns

    def _get_data_for_test(self, n_rows: Optional[int] = 5000):

        '''
        Get test klines data from local CSV file for testing purposes.

        NOTE: This is a test-only method used by SFDs to load sample data
        during test runs. Uses datasets/klines_2h_2020_2025.csv.

        Args:
            n_rows (int | None): Number of rows to read from CSV (default: 5000).
                                 If None, reads entire file.

        Returns:
            None (sets self.data with the loaded klines data)
        '''

        import pandas as pd

        df = pd.read_csv('datasets/klines_2h_2020_2025.csv', nrows=n_rows)
        df['datetime'] = pd.to_datetime(df['datetime'])
        self.data = pl.from_pandas(df)
        self.data_columns = self.data.columns
