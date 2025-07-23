'''
Test file for regime stability model
Location: test_regime_stability.py
'''

import loop
from loop.models.lightgbm import regime_stability
from loop.data import HistoricalData

# Test configuration constants
TEST_NUM_ROWS = 6000  # Number of rows to sample in tests
TEST_DATA_SIZE = 10000  # Total test data size
TEST_KLINE_SIZE = 7200  # 2 hour klines to match the model's interval
TEST_START_DATE = '2019-01-01'  # Start date matches notebook context


def test_regime_stability():
    '''Test regime stability model functionality.'''
    
    try:
        # Get historical data
        historical = HistoricalData()
        historical.get_historical_klines(
            n_rows=TEST_DATA_SIZE,
            kline_size=TEST_KLINE_SIZE,
            start_date_limit=TEST_START_DATE,
            futures=True
        )
        
        # Monkey patch the NUM_ROWS for testing
        import loop.models.lightgbm.regime_stability as rs
        original_num_rows = rs.NUM_ROWS
        rs.NUM_ROWS = TEST_NUM_ROWS
        
        try:
            # Initialize UEL with real historical data
            uel = loop.UniversalExperimentLoop(historical.data, rs)
            
            # Run single experiment
            uel.run(
                experiment_name="test_regime_stability",
                n_permutations=1,
                prep_each_round=False,
                random_search=True,
            )
            
            # Check results
            results = uel.log_df
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            
            # Check required columns exist
            required_cols = ['precision', 'recall', 'auc', 'accuracy']
            for col in required_cols:
                assert col in results.columns, f"Missing required column: {col}"
            
            # Check metrics are in valid ranges
            for col in required_cols:
                value = results[col][0]
                assert 0 <= value <= 1, f"{col} value out of range: {value}"
            
            print(f"\n✅ TEST PASSED")
            
            return True
            
        finally:
            # Restore original value
            rs.NUM_ROWS = original_num_rows
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_regime_stability()