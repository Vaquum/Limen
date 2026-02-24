import talib
import numpy as np
import time
from limen.data import HistoricalData
from limen.indicators import ad, adosc, atr, bbands, mfi, natr, obv, trange


historical = HistoricalData()
historical._get_data_for_test(n_rows=1000)
SAMPLE_DATA = historical.data

NUMPY_DATA = {
    'high': SAMPLE_DATA['high'].to_numpy(),
    'low': SAMPLE_DATA['low'].to_numpy(),
    'close': SAMPLE_DATA['close'].to_numpy(),
    'volume': SAMPLE_DATA['volume'].to_numpy()
}

TOLERANCE = 1e-8
DEFAULT_PERIOD = 14
FAST_PERIOD = 3
SLOW_PERIOD = 10
BB_WINDOW = 20
BB_NUM_STD = 2.0
T3_VFACTOR = 0.7


def test_ad():
    limen_result = ad(SAMPLE_DATA)['ad'].to_numpy()
    talib_result = talib.AD(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        NUMPY_DATA['volume'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_adosc():
    out_col = f'adosc_{FAST_PERIOD}_{SLOW_PERIOD}'
    limen_result = adosc(
        SAMPLE_DATA,
        fast_period=FAST_PERIOD,
        slow_period=SLOW_PERIOD,
    )[out_col].to_numpy()
    talib_result = talib.ADOSC(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        NUMPY_DATA['volume'],
        fastperiod=FAST_PERIOD,
        slowperiod=SLOW_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_mfi():
    out_col = f'mfi_{DEFAULT_PERIOD}'
    limen_result = mfi(SAMPLE_DATA, period=DEFAULT_PERIOD)[out_col].to_numpy()
    talib_result = talib.MFI(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        NUMPY_DATA['volume'],
        timeperiod=DEFAULT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_atr():
    out_col = f'atr_{DEFAULT_PERIOD}'
    limen_result = atr(SAMPLE_DATA, period=DEFAULT_PERIOD)[out_col].to_numpy()
    talib_result = talib.ATR(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        timeperiod=DEFAULT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_bbands():
    limen = bbands(
        SAMPLE_DATA,
        price_col='close',
        period=BB_WINDOW,
        nb_dev_up=BB_NUM_STD,
        nb_dev_dn=BB_NUM_STD,
        ma_type=0,
    )
    limen_upper = limen['bbands_upper'].to_numpy()
    limen_middle = limen['bbands_middle'].to_numpy()
    limen_lower = limen['bbands_lower'].to_numpy()

    talib_upper, talib_middle, talib_lower = talib.BBANDS(
        NUMPY_DATA['close'],
        timeperiod=BB_WINDOW,
        nbdevup=BB_NUM_STD,
        nbdevdn=BB_NUM_STD,
        matype=0,
    )

    verify_indicator_results_match(limen_upper, talib_upper)
    verify_indicator_results_match(limen_middle, talib_middle)
    verify_indicator_results_match(limen_lower, talib_lower)


def test_natr():
    out_col = f'natr_{DEFAULT_PERIOD}'
    limen_result = natr(SAMPLE_DATA, period=DEFAULT_PERIOD)[out_col].to_numpy()
    talib_result = talib.NATR(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        timeperiod=DEFAULT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_obv():
    limen_result = obv(SAMPLE_DATA)['obv'].to_numpy()
    talib_result = talib.OBV(
        NUMPY_DATA['close'],
        NUMPY_DATA['volume'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_trange():
    limen_result = trange(SAMPLE_DATA)['trange'].to_numpy()
    talib_result = talib.TRANGE(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def verify_indicator_results_match(limen_values, talib_values, tolerance=TOLERANCE):
    limen_values = np.asarray(limen_values, dtype=float)
    talib_values = np.asarray(talib_values, dtype=float)

    limen_clean = limen_values[~np.isnan(limen_values)]
    talib_clean = talib_values[~np.isnan(talib_values)]

    assert len(limen_clean) == len(talib_clean), (
        f"Length mismatch after NaN removal: "
        f"Limen={len(limen_clean)}, TA-Lib={len(talib_clean)}"
    )

    assert len(limen_clean) > 0, "No valid (non-NaN) values to compare"

    diff = np.abs(limen_clean - talib_clean)
    max_diff = np.max(diff)

    print(
        f"First 10 valid values -\n"
        f"\tLimen: {limen_clean[:10]}\n"
        f"\tTA-Lib: {talib_clean[:10]}"
    )
    print(
        f"Last 10 valid values -\n"
        f"\tLimen: {limen_clean[-10:]}\n"
        f"\tTA-Lib: {talib_clean[-10:]}"
    )

    assert max_diff < tolerance, f"Results differ, Max Diff: {max_diff}"


def test_indicators_vs_talib():
    test_functions = [
        ('AD', test_ad),
        ('ADOSC', test_adosc),
        ('ATR', test_atr),
        ('BBANDS', test_bbands),
        ('MFI', test_mfi),
        ('NATR', test_natr),
        ('OBV', test_obv),
        ('TRANGE', test_trange),
    ]

    print("Running indicator validation tests against TA-Lib...")
    print("=" * 60)

    total_time = 0.0
    test_results = []

    for indicator_name, test_func in test_functions:
        start_time = time.time()
        try:
            test_func()
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            test_results.append((indicator_name, "PASSED", execution_time))
            print(f'    ✅ {indicator_name}: PASSED ({execution_time:.4f}s)')
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            test_results.append((indicator_name, f"FAILED - {e}", execution_time))
            print(f'    ❌ {indicator_name}: FAILED - {e} ({execution_time:.4f}s)')

    print("=" * 60)
    print(f"Total execution time: {total_time:.4f}s")
    print(f"Average time per test: {total_time/len(test_functions):.4f}s")
    print(f"Tests completed: {len([r for r in test_results if r[1] == 'PASSED'])}/{len(test_results)} passed")


if __name__ == "__main__":
    test_indicators_vs_talib()
