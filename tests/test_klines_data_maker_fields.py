from limen.data import HistoricalData

def test_klines_data_maker_fields():

    historical = HistoricalData()
    historical._get_data_for_test(n_rows=None)
    df = historical.data
    
    required_columns = [
        'datetime', 'open', 'high', 'low', 'close', 'mean', 'std', 'median', 'iqr',
        'volume', 'maker_ratio', 'no_of_trades', 'open_liquidity', 'high_liquidity',
        'low_liquidity', 'close_liquidity', 'liquidity_sum', 'maker_volume', 'maker_liquidity'
    ]
    
    assert len(df.columns) == len(required_columns)
    
    for col in required_columns:
        assert col in df.columns
    
    assert 'maker_volume' in df.columns
    assert 'maker_liquidity' in df.columns
    
    assert df['maker_volume'].null_count() == 0
    assert df['maker_liquidity'].null_count() == 0
    
    assert (df['maker_volume'] >= 0).all()
    assert (df['maker_liquidity'] >= 0).all()
    
    assert (df['maker_volume'] <= df['volume']).all()
    assert (df['maker_liquidity'] <= df['liquidity_sum']).all()

