import talib
import numpy as np
import time
from limen.data import HistoricalData
from limen.indicators import ad, adosc, apo, atr, bbands, bop, cci, cmo, dema, ema, ht_trendline, kama, ma, macd, macdfix, macdext, mama, mfi, mom, natr, obv, ppo, sar, sarext, sma, t3, tema, trange, trima, tsf, wma


historical = HistoricalData()
historical._get_data_for_test(n_rows=1000)
SAMPLE_DATA = historical.data

NUMPY_DATA = {
    'open': SAMPLE_DATA['open'].to_numpy(),
    'high': SAMPLE_DATA['high'].to_numpy(),
    'low': SAMPLE_DATA['low'].to_numpy(),
    'close': SAMPLE_DATA['close'].to_numpy(),
    'volume': SAMPLE_DATA['volume'].to_numpy()
}

TOLERANCE = 1e-8
DEFAULT_PERIOD = 14
DEMA_PERIOD = 30
EMA_PERIOD = 30
KAMA_PERIOD = 30
MA_PERIOD = 30
MA_TYPES = list(range(9))
SMA_PERIOD = 30
TEMA_PERIOD = 30
TRIMA_PERIOD = 30
TSF_PERIOD = 14
WMA_PERIOD = 30
MAMA_FAST_LIMIT = 0.5
MAMA_SLOW_LIMIT = 0.05
SAR_ACCELERATION = 0.02
SAR_MAXIMUM = 0.2
SAREXT_START_VALUE = 0.0
SAREXT_OFFSET_ON_REVERSE = 0.0
SAREXT_AF_INIT_LONG = 0.02
SAREXT_AF_LONG = 0.02
SAREXT_AF_MAX_LONG = 0.2
SAREXT_AF_INIT_SHORT = 0.02
SAREXT_AF_SHORT = 0.02
SAREXT_AF_MAX_SHORT = 0.2
FAST_PERIOD = 3
SLOW_PERIOD = 10
APO_FAST_PERIOD = 12
APO_SLOW_PERIOD = 26
PPO_FAST_PERIOD = 12
PPO_SLOW_PERIOD = 26
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
MACDFIX_SIGNAL_PERIOD = 9
MOM_PERIOD = 10
BB_WINDOW = 20
BB_NUM_STD = 2.0
BB_MA_TYPES = list(range(9))
T3_VFACTOR = 0.7
T3_PERIOD = 5
MACDEXT_CASES = [
    (12, 0, 26, 0, 9, 0),
    (12, 1, 26, 1, 9, 1),
    (8, 2, 21, 5, 7, 0),
]


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


def test_apo():
    for ma_type in MA_TYPES:
        out_col = f'apo_{APO_FAST_PERIOD}_{APO_SLOW_PERIOD}_{ma_type}'
        limen_result = apo(
            SAMPLE_DATA,
            price_col='close',
            fast_period=APO_FAST_PERIOD,
            slow_period=APO_SLOW_PERIOD,
            ma_type=ma_type,
        )[out_col].to_numpy()
        talib_result = talib.APO(
            NUMPY_DATA['close'],
            fastperiod=APO_FAST_PERIOD,
            slowperiod=APO_SLOW_PERIOD,
            matype=ma_type,
        )

        verify_indicator_results_match(limen_result, talib_result)


def test_ppo():
    for ma_type in MA_TYPES:
        out_col = f'ppo_{PPO_FAST_PERIOD}_{PPO_SLOW_PERIOD}_{ma_type}'
        limen_result = ppo(
            SAMPLE_DATA,
            price_col='close',
            fast_period=PPO_FAST_PERIOD,
            slow_period=PPO_SLOW_PERIOD,
            ma_type=ma_type,
        )[out_col].to_numpy()
        talib_result = talib.PPO(
            NUMPY_DATA['close'],
            fastperiod=PPO_FAST_PERIOD,
            slowperiod=PPO_SLOW_PERIOD,
            matype=ma_type,
        )

        verify_indicator_results_match(limen_result, talib_result)


def test_dema():
    out_col = f'dema_{DEMA_PERIOD}'
    limen_result = dema(SAMPLE_DATA, price_col='close', period=DEMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.DEMA(
        NUMPY_DATA['close'],
        timeperiod=DEMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_ema():
    out_col = f'ema_{EMA_PERIOD}'
    limen_result = ema(SAMPLE_DATA, price_col='close', period=EMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.EMA(
        NUMPY_DATA['close'],
        timeperiod=EMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_ht_trendline():
    limen_result = ht_trendline(SAMPLE_DATA, price_col='close')['ht_trendline'].to_numpy()
    talib_result = talib.HT_TRENDLINE(NUMPY_DATA['close'])

    verify_indicator_results_match(limen_result, talib_result)


def test_kama():
    out_col = f'kama_{KAMA_PERIOD}'
    limen_result = kama(SAMPLE_DATA, price_col='close', period=KAMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.KAMA(
        NUMPY_DATA['close'],
        timeperiod=KAMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_ma():
    for ma_type in MA_TYPES:
        out_col = f'ma_{MA_PERIOD}_{ma_type}'
        limen_result = ma(
            SAMPLE_DATA,
            price_col='close',
            period=MA_PERIOD,
            ma_type=ma_type,
        )[out_col].to_numpy()
        talib_result = talib.MA(
            NUMPY_DATA['close'],
            timeperiod=MA_PERIOD,
            matype=ma_type,
        )

        verify_indicator_results_match(limen_result, talib_result)


def test_mama():
    limen = mama(
        SAMPLE_DATA,
        price_col='close',
        fast_limit=MAMA_FAST_LIMIT,
        slow_limit=MAMA_SLOW_LIMIT,
    )
    limen_mama = limen['mama'].to_numpy()
    limen_fama = limen['fama'].to_numpy()

    talib_mama, talib_fama = talib.MAMA(
        NUMPY_DATA['close'],
        fastlimit=MAMA_FAST_LIMIT,
        slowlimit=MAMA_SLOW_LIMIT,
    )

    verify_indicator_results_match(limen_mama, talib_mama)
    verify_indicator_results_match(limen_fama, talib_fama)


def test_sma():
    out_col = f'sma_{SMA_PERIOD}'
    limen_result = sma(SAMPLE_DATA, price_col='close', period=SMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.SMA(
        NUMPY_DATA['close'],
        timeperiod=SMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_tema():
    out_col = f'tema_{TEMA_PERIOD}'
    limen_result = tema(SAMPLE_DATA, price_col='close', period=TEMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.TEMA(
        NUMPY_DATA['close'],
        timeperiod=TEMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_trima():
    out_col = f'trima_{TRIMA_PERIOD}'
    limen_result = trima(SAMPLE_DATA, price_col='close', period=TRIMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.TRIMA(
        NUMPY_DATA['close'],
        timeperiod=TRIMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_t3():
    out_col = f't3_{T3_PERIOD}_{T3_VFACTOR:g}'
    limen_result = t3(
        SAMPLE_DATA,
        price_col='close',
        period=T3_PERIOD,
        vfactor=T3_VFACTOR,
    )[out_col].to_numpy()
    talib_result = talib.T3(
        NUMPY_DATA['close'],
        timeperiod=T3_PERIOD,
        vfactor=T3_VFACTOR,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_tsf():
    out_col = f'tsf_{TSF_PERIOD}'
    limen_result = tsf(SAMPLE_DATA, price_col='close', period=TSF_PERIOD)[out_col].to_numpy()
    talib_result = talib.TSF(
        NUMPY_DATA['close'],
        timeperiod=TSF_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_wma():
    out_col = f'wma_{WMA_PERIOD}'
    limen_result = wma(SAMPLE_DATA, price_col='close', period=WMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.WMA(
        NUMPY_DATA['close'],
        timeperiod=WMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_sar():
    limen_result = sar(
        SAMPLE_DATA,
        high_col='high',
        low_col='low',
        acceleration=SAR_ACCELERATION,
        maximum=SAR_MAXIMUM,
    )['sar'].to_numpy()
    talib_result = talib.SAR(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        acceleration=SAR_ACCELERATION,
        maximum=SAR_MAXIMUM,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_sarext():
    limen_result = sarext(
        SAMPLE_DATA,
        high_col='high',
        low_col='low',
        start_value=SAREXT_START_VALUE,
        offset_on_reverse=SAREXT_OFFSET_ON_REVERSE,
        acceleration_init_long=SAREXT_AF_INIT_LONG,
        acceleration_long=SAREXT_AF_LONG,
        acceleration_max_long=SAREXT_AF_MAX_LONG,
        acceleration_init_short=SAREXT_AF_INIT_SHORT,
        acceleration_short=SAREXT_AF_SHORT,
        acceleration_max_short=SAREXT_AF_MAX_SHORT,
    )['sarext'].to_numpy()
    talib_result = talib.SAREXT(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        startvalue=SAREXT_START_VALUE,
        offsetonreverse=SAREXT_OFFSET_ON_REVERSE,
        accelerationinitlong=SAREXT_AF_INIT_LONG,
        accelerationlong=SAREXT_AF_LONG,
        accelerationmaxlong=SAREXT_AF_MAX_LONG,
        accelerationinitshort=SAREXT_AF_INIT_SHORT,
        accelerationshort=SAREXT_AF_SHORT,
        accelerationmaxshort=SAREXT_AF_MAX_SHORT,
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
    for ma_type in BB_MA_TYPES:
        limen = bbands(
            SAMPLE_DATA,
            price_col='close',
            period=BB_WINDOW,
            nb_dev_up=BB_NUM_STD,
            nb_dev_dn=BB_NUM_STD,
            ma_type=ma_type,
        )
        limen_upper = limen['bbands_upper'].to_numpy()
        limen_middle = limen['bbands_middle'].to_numpy()
        limen_lower = limen['bbands_lower'].to_numpy()

        talib_upper, talib_middle, talib_lower = talib.BBANDS(
            NUMPY_DATA['close'],
            timeperiod=BB_WINDOW,
            nbdevup=BB_NUM_STD,
            nbdevdn=BB_NUM_STD,
            matype=ma_type,
        )

        verify_indicator_results_match(limen_upper, talib_upper)
        verify_indicator_results_match(limen_middle, talib_middle)
        verify_indicator_results_match(limen_lower, talib_lower)


def test_bop():
    limen_result = bop(SAMPLE_DATA)['bop'].to_numpy()
    talib_result = talib.BOP(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cci():
    out_col = f'cci_{DEFAULT_PERIOD}'
    limen_result = cci(SAMPLE_DATA, period=DEFAULT_PERIOD)[out_col].to_numpy()
    talib_result = talib.CCI(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        timeperiod=DEFAULT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cmo():
    out_col = f'cmo_{DEFAULT_PERIOD}'
    limen_result = cmo(SAMPLE_DATA, price_col='close', period=DEFAULT_PERIOD)[out_col].to_numpy()
    talib_result = talib.CMO(
        NUMPY_DATA['close'],
        timeperiod=DEFAULT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_macd():
    limen = macd(
        SAMPLE_DATA,
        price_col='close',
        fast_period=MACD_FAST_PERIOD,
        slow_period=MACD_SLOW_PERIOD,
        signal_period=MACD_SIGNAL_PERIOD,
    )
    limen_macd = limen['macd'].to_numpy()
    limen_signal = limen['macd_signal'].to_numpy()
    limen_hist = limen['macd_hist'].to_numpy()

    talib_macd, talib_signal, talib_hist = talib.MACD(
        NUMPY_DATA['close'],
        fastperiod=MACD_FAST_PERIOD,
        slowperiod=MACD_SLOW_PERIOD,
        signalperiod=MACD_SIGNAL_PERIOD,
    )

    verify_indicator_results_match(limen_macd, talib_macd)
    verify_indicator_results_match(limen_signal, talib_signal)
    verify_indicator_results_match(limen_hist, talib_hist)


def test_macdfix():
    limen = macdfix(
        SAMPLE_DATA,
        price_col='close',
        signal_period=MACDFIX_SIGNAL_PERIOD,
    )
    limen_macd = limen['macdfix'].to_numpy()
    limen_signal = limen['macdfix_signal'].to_numpy()
    limen_hist = limen['macdfix_hist'].to_numpy()

    talib_macd, talib_signal, talib_hist = talib.MACDFIX(
        NUMPY_DATA['close'],
        signalperiod=MACDFIX_SIGNAL_PERIOD,
    )

    verify_indicator_results_match(limen_macd, talib_macd)
    verify_indicator_results_match(limen_signal, talib_signal)
    verify_indicator_results_match(limen_hist, talib_hist)


def test_macdext():
    for fast_p, fast_t, slow_p, slow_t, signal_p, signal_t in MACDEXT_CASES:
        limen = macdext(
            SAMPLE_DATA,
            price_col='close',
            fast_period=fast_p,
            fast_ma_type=fast_t,
            slow_period=slow_p,
            slow_ma_type=slow_t,
            signal_period=signal_p,
            signal_ma_type=signal_t,
        )
        limen_macd = limen['macdext'].to_numpy()
        limen_signal = limen['macdext_signal'].to_numpy()
        limen_hist = limen['macdext_hist'].to_numpy()

        talib_macd, talib_signal, talib_hist = talib.MACDEXT(
            NUMPY_DATA['close'],
            fastperiod=fast_p,
            fastmatype=fast_t,
            slowperiod=slow_p,
            slowmatype=slow_t,
            signalperiod=signal_p,
            signalmatype=signal_t,
        )

        verify_indicator_results_match(limen_macd, talib_macd)
        verify_indicator_results_match(limen_signal, talib_signal)
        verify_indicator_results_match(limen_hist, talib_hist)


def test_mom():
    out_col = f'mom_{MOM_PERIOD}'
    limen_result = mom(SAMPLE_DATA, price_col='close', period=MOM_PERIOD)[out_col].to_numpy()
    talib_result = talib.MOM(
        NUMPY_DATA['close'],
        timeperiod=MOM_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


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
        ('APO', test_apo),
        ('PPO', test_ppo),
        ('DEMA', test_dema),
        ('EMA', test_ema),
        ('HT_TRENDLINE', test_ht_trendline),
        ('KAMA', test_kama),
        ('MA', test_ma),
        ('MAMA', test_mama),
        ('SMA', test_sma),
        ('TEMA', test_tema),
        ('TRIMA', test_trima),
        ('T3', test_t3),
        ('TSF', test_tsf),
        ('WMA', test_wma),
        ('SAR', test_sar),
        ('SAREXT', test_sarext),
        ('ATR', test_atr),
        ('BBANDS', test_bbands),
        ('BOP', test_bop),
        ('CCI', test_cci),
        ('CMO', test_cmo),
        ('MACD', test_macd),
        ('MACDFIX', test_macdfix),
        ('MACDEXT', test_macdext),
        ('MOM', test_mom),
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
