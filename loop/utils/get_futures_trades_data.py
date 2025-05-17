from typing import Optional, Tuple
from .generic_endpoint_for_tdw import generic_endpoint_for_tdw
import polars as pl


def get_futures_trades_data(month_year: Optional[Tuple[int,int]] = None,
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

    return generic_endpoint_for_tdw(month_year,
                                    n_rows,
                                    include_datetime_col,
                                    show_summary,
                                    select_cols,
                                    table_name,
                                    sort_by)
