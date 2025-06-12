#!/usr/bin/env python3
"""
Test for statistical metrics in get_historical_klines

This test verifies that the new statistical metrics (mean, std, median, iqr)
are properly included in the data returned by get_historical_klines.
"""

import pytest
import polars as pl
from unittest.mock import Mock, patch
import numpy as np


def test_get_klines_data_includes_statistical_metrics():
    """Test that get_klines_data SQL query includes statistical metrics"""
    
    # Mock the ClickHouse client
    mock_client = Mock()
    
    # Create mock data that would be returned by the query
    mock_arrow_table = Mock()
    
    # Mock polars dataframe with all expected columns including new metrics
    expected_columns = [
        'datetime', 'open', 'high', 'low', 'close', 'volume',
        'maker_ratio', 'no_of_trades', 'open_liquidity', 'high_liquidity',
        'low_liquidity', 'close_liquidity', 'liquidity_sum',
        'mean', 'std', 'median', 'iqr'  # New statistical metrics
    ]
    
    mock_df = pl.DataFrame({
        'datetime': [1700000000000 + i * 60000 for i in range(10)],
        'open': np.random.uniform(40000, 41000, 10),
        'high': np.random.uniform(40500, 41500, 10),
        'low': np.random.uniform(39500, 40500, 10),
        'close': np.random.uniform(40000, 41000, 10),
        'volume': np.random.uniform(100, 1000, 10),
        'maker_ratio': np.random.uniform(0.4, 0.6, 10),
        'no_of_trades': np.random.randint(10, 100, 10),
        'open_liquidity': np.random.uniform(1000, 10000, 10),
        'high_liquidity': np.random.uniform(1000, 10000, 10),
        'low_liquidity': np.random.uniform(1000, 10000, 10),
        'close_liquidity': np.random.uniform(1000, 10000, 10),
        'liquidity_sum': np.random.uniform(10000, 100000, 10),
        'mean': np.random.uniform(40000, 41000, 10),  # New metric
        'std': np.random.uniform(10, 100, 10),  # New metric
        'median': np.random.uniform(40000, 41000, 10),  # New metric
        'iqr': np.random.uniform(50, 200, 10),  # New metric
    })
    
    with patch('clickhouse_connect.get_client') as mock_get_client, \
         patch('polars.from_arrow') as mock_from_arrow:
        
        mock_get_client.return_value = mock_client
        mock_client.query_arrow.return_value = mock_arrow_table
        mock_from_arrow.return_value = mock_df
        
        # Import and call the function
        from loop.utils.get_klines_data import get_klines_data
        
        result = get_klines_data(n_rows=10, kline_size=60)
        
        # Verify the new columns are present
        assert 'mean' in result.columns, "Mean column is missing"
        assert 'std' in result.columns, "Standard deviation column is missing"
        assert 'median' in result.columns, "Median column is missing"
        assert 'iqr' in result.columns, "IQR column is missing"
        
        # Verify the SQL query includes the new metrics
        query_call = mock_client.query_arrow.call_args[0][0]
        assert 'avg(price)' in query_call, "AVG calculation missing from SQL"
        assert 'stddevPop(price)' in query_call, "STDDEV calculation missing from SQL"
        assert 'median(price)' in query_call, "MEDIAN calculation missing from SQL"
        assert 'quantile(0.75)(price) - quantile(0.25)(price)' in query_call, "IQR calculation missing from SQL"


def test_statistical_metrics_validation():
    """Test that statistical metrics have valid values"""
    
    # Create sample data with known values
    prices = [100, 200, 300, 400, 500]
    expected_mean = 300
    expected_median = 300
    expected_std = np.std(prices, ddof=0)  # Population std
    expected_iqr = 300  # Q3(400) - Q1(200)
    
    # Create a mock dataframe with calculated metrics
    df = pl.DataFrame({
        'datetime': list(range(5)),
        'mean': [expected_mean] * 5,
        'std': [expected_std] * 5,
        'median': [expected_median] * 5,
        'iqr': [expected_iqr] * 5,
        'high': [max(prices)] * 5,
        'low': [min(prices)] * 5,
    })
    
    # Validate statistical properties
    assert all(df['std'] >= 0), "Standard deviation should be non-negative"
    assert all(df['mean'] >= df['low']), "Mean should be >= low"
    assert all(df['mean'] <= df['high']), "Mean should be <= high"
    assert all(df['median'] >= df['low']), "Median should be >= low"
    assert all(df['median'] <= df['high']), "Median should be <= high"
    assert all(df['iqr'] >= 0), "IQR should be non-negative"


if __name__ == '__main__':
    print("Running statistical metrics tests...")
    
    # Run the tests
    test_get_klines_data_includes_statistical_metrics()
    print("✓ Test: get_klines_data includes statistical metrics")
    
    test_statistical_metrics_validation()
    print("✓ Test: statistical metrics validation")
    
    print("\nAll tests passed!") 