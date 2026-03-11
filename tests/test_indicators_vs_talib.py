import talib
import numpy as np
from limen.data import HistoricalData
from limen.indicators import ad, adosc, apo, atr, avgprice, bbands, bop, cci, cdladvancedblock, cdlabandonedbaby, cdlbelthold, cdlconcealbabyswall, cdlclosingmarubozu, cdlcounterattack, cdldarkcloudcover, cdldragonflydoji, cdlengulfing, cdlgravestonedoji, cdlhammer, cdlhangingman, cdlharami, cdlharamicross, cdlhighwave, cdlhikkake, cdlhikkakemod, cdlhomingpigeon, cdlidentical3crows, cdlinvertedhammer, cdlladderbottom, cdllongleggeddoji, cdllongline, cdlmarubozu, cdlmatchinglow, cdlmathold, cdlonneck, cdlpiercing, cdlrickshawman, cdlrisefall3methods, cdlseparatinglines, cdlshootingstar, cdlshortline, cdlspinningtop, cdlstalledpattern, cdlsticksandwich, cdltakuri, cdlthrusting, cdltristar, cdlunique3river, cdl2crows, cdl3blackcrows, cdl3inside, cdl3linestrike, cdl3starsinsouth, cdl3whitesoldiers, coldoji, cmo, dema, ema, ht_dcperiod, ht_dcphase, ht_phasor, ht_sine, ht_trendline, ht_trendmode, kama, linearreg, linearreg_angle, linearreg_intercept, linearreg_slope, ma, macd, macdfix, macdext, mama, medprice, midpoint, midprice, mfi, mom, natr, obv, ppo, roc, rocp, rocr, rocr100, rsi, sar, sarext, sma, stddev, stoch, stochf, stochrsi, t3, tema, trange, trima, trix, tsf, typprice, ultosc, var, wclprice, willr, wma


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

TOLERANCE = 1e-7
DEFAULT_PERIOD = 14
DEMA_PERIOD = 30
EMA_PERIOD = 30
KAMA_PERIOD = 30
MA_PERIOD = 30
MIDPOINT_PERIOD = 14
MIDPRICE_PERIOD = 14
MA_TYPES = list(range(9))
SMA_PERIOD = 30
TEMA_PERIOD = 30
TRIMA_PERIOD = 30
TRIX_PERIOD = 30
TSF_PERIOD = 14
LINEARREG_PERIOD = 14
LINEARREG_ANGLE_PERIOD = 14
LINEARREG_INTERCEPT_PERIOD = 14
LINEARREG_SLOPE_PERIOD = 14
STDDEV_PERIOD = 5
STDDEV_NBDEV = 1.0
VAR_PERIOD = 5
VAR_NBDEV = 1.0
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
ROC_PERIOD = 10
ROCP_PERIOD = 10
ROCR_PERIOD = 10
ROCR100_PERIOD = 10
RSI_PERIOD = 14
STOCH_FASTK_PERIOD = 5
STOCH_SLOWK_PERIOD = 3
STOCH_SLOWD_PERIOD = 3
STOCHF_FASTK_PERIOD = 5
STOCHF_FASTD_PERIOD = 3
STOCHRSI_PERIOD = 14
STOCHRSI_FASTK_PERIOD = 5
STOCHRSI_FASTD_PERIOD = 3
ULTOSC_PERIOD1 = 7
ULTOSC_PERIOD2 = 14
ULTOSC_PERIOD3 = 28
WILLR_PERIOD = 14
BB_WINDOW = 20
BB_NUM_STD = 2.0
BB_MA_TYPES = list(range(9))
T3_VFACTOR = 0.7
T3_PERIOD = 5
ABANDONEDBABY_PENETRATION = 0.3
DARKCLOUDCOVER_PENETRATION = 0.5
MATHOLD_PENETRATION = 0.5
MACDEXT_CASES = list(dict.fromkeys(
    [(12, ma_type, 26, 0, 9, 0) for ma_type in MA_TYPES]
    + [(12, 0, 26, ma_type, 9, 0) for ma_type in MA_TYPES]
    + [(12, 0, 26, 0, 9, ma_type) for ma_type in MA_TYPES]
))


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
    out_col = f"adosc_{FAST_PERIOD}_{SLOW_PERIOD}"
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
        out_col = f"apo_{APO_FAST_PERIOD}_{APO_SLOW_PERIOD}_{ma_type}"
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
        out_col = f"ppo_{PPO_FAST_PERIOD}_{PPO_SLOW_PERIOD}_{ma_type}"
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
    out_col = f"dema_{DEMA_PERIOD}"
    limen_result = dema(SAMPLE_DATA, price_col='close', period=DEMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.DEMA(
        NUMPY_DATA['close'],
        timeperiod=DEMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_ema():
    out_col = f"ema_{EMA_PERIOD}"
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


def test_ht_dcperiod():
    limen_result = ht_dcperiod(SAMPLE_DATA, price_col='close')['ht_dcperiod'].to_numpy()
    talib_result = talib.HT_DCPERIOD(NUMPY_DATA['close'])

    verify_indicator_results_match(limen_result, talib_result)


def test_ht_dcphase():
    limen_result = ht_dcphase(SAMPLE_DATA, price_col='close')['ht_dcphase'].to_numpy()
    talib_result = talib.HT_DCPHASE(NUMPY_DATA['close'])

    verify_indicator_results_match(limen_result, talib_result)


def test_ht_phasor():
    limen = ht_phasor(SAMPLE_DATA, price_col='close')
    limen_inphase = limen['ht_phasor_inphase'].to_numpy()
    limen_quadrature = limen['ht_phasor_quadrature'].to_numpy()

    talib_inphase, talib_quadrature = talib.HT_PHASOR(NUMPY_DATA['close'])

    verify_indicator_results_match(limen_inphase, talib_inphase)
    verify_indicator_results_match(limen_quadrature, talib_quadrature)


def test_ht_sine():
    limen = ht_sine(SAMPLE_DATA, price_col='close')
    limen_sine = limen['ht_sine'].to_numpy()
    limen_lead_sine = limen['ht_sine_lead'].to_numpy()

    talib_sine, talib_lead_sine = talib.HT_SINE(NUMPY_DATA['close'])

    verify_indicator_results_match(limen_sine, talib_sine)
    verify_indicator_results_match(limen_lead_sine, talib_lead_sine)


def test_ht_trendmode():
    limen_result = ht_trendmode(SAMPLE_DATA, price_col='close')['ht_trendmode'].to_numpy()
    talib_result = talib.HT_TRENDMODE(NUMPY_DATA['close'])

    verify_indicator_results_match(limen_result, talib_result)


def test_kama():
    out_col = f"kama_{KAMA_PERIOD}"
    limen_result = kama(SAMPLE_DATA, price_col='close', period=KAMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.KAMA(
        NUMPY_DATA['close'],
        timeperiod=KAMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_ma():
    for ma_type in MA_TYPES:
        out_col = f"ma_{MA_PERIOD}_{ma_type}"
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
    out_col = f"sma_{SMA_PERIOD}"
    limen_result = sma(SAMPLE_DATA, price_col='close', period=SMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.SMA(
        NUMPY_DATA['close'],
        timeperiod=SMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_tema():
    out_col = f"tema_{TEMA_PERIOD}"
    limen_result = tema(SAMPLE_DATA, price_col='close', period=TEMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.TEMA(
        NUMPY_DATA['close'],
        timeperiod=TEMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_trima():
    out_col = f"trima_{TRIMA_PERIOD}"
    limen_result = trima(SAMPLE_DATA, price_col='close', period=TRIMA_PERIOD)[out_col].to_numpy()
    talib_result = talib.TRIMA(
        NUMPY_DATA['close'],
        timeperiod=TRIMA_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_trix():
    out_col = f"trix_{TRIX_PERIOD}"
    limen_result = trix(SAMPLE_DATA, price_col='close', period=TRIX_PERIOD)[out_col].to_numpy()
    talib_result = talib.TRIX(
        NUMPY_DATA['close'],
        timeperiod=TRIX_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_t3():
    out_col = f"t3_{T3_PERIOD}_{T3_VFACTOR:g}"
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
    out_col = f"tsf_{TSF_PERIOD}"
    limen_result = tsf(SAMPLE_DATA, price_col='close', period=TSF_PERIOD)[out_col].to_numpy()
    talib_result = talib.TSF(
        NUMPY_DATA['close'],
        timeperiod=TSF_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_linearreg():
    out_col = f"linearreg_{LINEARREG_PERIOD}"
    limen_result = linearreg(SAMPLE_DATA, price_col='close', period=LINEARREG_PERIOD)[out_col].to_numpy()
    talib_result = talib.LINEARREG(
        NUMPY_DATA['close'],
        timeperiod=LINEARREG_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_linearreg_angle():
    out_col = f"linearreg_angle_{LINEARREG_ANGLE_PERIOD}"
    limen_result = linearreg_angle(SAMPLE_DATA, price_col='close', period=LINEARREG_ANGLE_PERIOD)[out_col].to_numpy()
    talib_result = talib.LINEARREG_ANGLE(
        NUMPY_DATA['close'],
        timeperiod=LINEARREG_ANGLE_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_linearreg_intercept():
    out_col = f"linearreg_intercept_{LINEARREG_INTERCEPT_PERIOD}"
    limen_result = linearreg_intercept(SAMPLE_DATA, price_col='close', period=LINEARREG_INTERCEPT_PERIOD)[out_col].to_numpy()
    talib_result = talib.LINEARREG_INTERCEPT(
        NUMPY_DATA['close'],
        timeperiod=LINEARREG_INTERCEPT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_linearreg_slope():
    out_col = f"linearreg_slope_{LINEARREG_SLOPE_PERIOD}"
    limen_result = linearreg_slope(SAMPLE_DATA, price_col='close', period=LINEARREG_SLOPE_PERIOD)[out_col].to_numpy()
    talib_result = talib.LINEARREG_SLOPE(
        NUMPY_DATA['close'],
        timeperiod=LINEARREG_SLOPE_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_stddev():
    out_col = f"stddev_{STDDEV_PERIOD}_{STDDEV_NBDEV:g}"
    limen_result = stddev(
        SAMPLE_DATA,
        price_col='close',
        period=STDDEV_PERIOD,
        nb_dev=STDDEV_NBDEV,
    )[out_col].to_numpy()
    talib_result = talib.STDDEV(
        NUMPY_DATA['close'],
        timeperiod=STDDEV_PERIOD,
        nbdev=STDDEV_NBDEV,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_var():
    out_col = f"var_{VAR_PERIOD}_{VAR_NBDEV:g}"
    limen_result = var(
        SAMPLE_DATA,
        price_col='close',
        period=VAR_PERIOD,
        nb_dev=VAR_NBDEV,
    )[out_col].to_numpy()
    talib_result = talib.VAR(
        NUMPY_DATA['close'],
        timeperiod=VAR_PERIOD,
        nbdev=VAR_NBDEV,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_wma():
    out_col = f"wma_{WMA_PERIOD}"
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
    out_col = f"mfi_{DEFAULT_PERIOD}"
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
    out_col = f"atr_{DEFAULT_PERIOD}"
    limen_result = atr(SAMPLE_DATA, period=DEFAULT_PERIOD)[out_col].to_numpy()
    talib_result = talib.ATR(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        timeperiod=DEFAULT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_avgprice():
    limen_result = avgprice(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['avgprice'].to_numpy()
    talib_result = talib.AVGPRICE(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_medprice():
    limen_result = medprice(
        SAMPLE_DATA,
        high_col='high',
        low_col='low',
    )['medprice'].to_numpy()
    talib_result = talib.MEDPRICE(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_midpoint():
    out_col = f"midpoint_{MIDPOINT_PERIOD}"
    limen_result = midpoint(SAMPLE_DATA, price_col='close', period=MIDPOINT_PERIOD)[out_col].to_numpy()
    talib_result = talib.MIDPOINT(
        NUMPY_DATA['close'],
        timeperiod=MIDPOINT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_midprice():
    out_col = f"midprice_{MIDPRICE_PERIOD}"
    limen_result = midprice(
        SAMPLE_DATA,
        high_col='high',
        low_col='low',
        period=MIDPRICE_PERIOD,
    )[out_col].to_numpy()
    talib_result = talib.MIDPRICE(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        timeperiod=MIDPRICE_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_typprice():
    limen_result = typprice(
        SAMPLE_DATA,
        high_col='high',
        low_col='low',
        close_col='close',
    )['typprice'].to_numpy()
    talib_result = talib.TYPPRICE(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_wclprice():
    limen_result = wclprice(
        SAMPLE_DATA,
        high_col='high',
        low_col='low',
        close_col='close',
    )['wclprice'].to_numpy()
    talib_result = talib.WCLPRICE(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdl2crows():
    limen_result = cdl2crows(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdl2crows'].to_numpy()
    talib_result = talib.CDL2CROWS(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdl3blackcrows():
    limen_result = cdl3blackcrows(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdl3blackcrows'].to_numpy()
    talib_result = talib.CDL3BLACKCROWS(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdl3inside():
    limen_result = cdl3inside(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdl3inside'].to_numpy()
    talib_result = talib.CDL3INSIDE(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdl3linestrike():
    limen_result = cdl3linestrike(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdl3linestrike'].to_numpy()
    talib_result = talib.CDL3LINESTRIKE(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdl3starsinsouth():
    limen_result = cdl3starsinsouth(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdl3starsinsouth'].to_numpy()
    talib_result = talib.CDL3STARSINSOUTH(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdl3whitesoldiers():
    limen_result = cdl3whitesoldiers(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdl3whitesoldiers'].to_numpy()
    talib_result = talib.CDL3WHITESOLDIERS(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlabandonedbaby():
    limen_result = cdlabandonedbaby(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
        penetration=ABANDONEDBABY_PENETRATION,
    )['cdlabandonedbaby'].to_numpy()
    talib_result = talib.CDLABANDONEDBABY(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        penetration=ABANDONEDBABY_PENETRATION,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdladvancedblock():
    limen_result = cdladvancedblock(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdladvancedblock'].to_numpy()
    talib_result = talib.CDLADVANCEBLOCK(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlbelthold():
    limen_result = cdlbelthold(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlbelthold'].to_numpy()
    talib_result = talib.CDLBELTHOLD(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlclosingmarubozu():
    limen_result = cdlclosingmarubozu(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlclosingmarubozu'].to_numpy()
    talib_result = talib.CDLCLOSINGMARUBOZU(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlconcealbabyswall():
    limen_result = cdlconcealbabyswall(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlconcealbabyswall'].to_numpy()
    talib_result = talib.CDLCONCEALBABYSWALL(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlcounterattack():
    limen_result = cdlcounterattack(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlcounterattack'].to_numpy()
    talib_result = talib.CDLCOUNTERATTACK(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdldarkcloudcover():
    limen_result = cdldarkcloudcover(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
        penetration=DARKCLOUDCOVER_PENETRATION,
    )['cdldarkcloudcover'].to_numpy()
    talib_result = talib.CDLDARKCLOUDCOVER(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        penetration=DARKCLOUDCOVER_PENETRATION,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_coldoji():
    limen_result = coldoji(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['coldoji'].to_numpy()
    talib_result = talib.CDLDOJI(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdldragonflydoji():
    limen_result = cdldragonflydoji(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdldragonflydoji'].to_numpy()
    talib_result = talib.CDLDRAGONFLYDOJI(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlengulfing():
    limen_result = cdlengulfing(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlengulfing'].to_numpy()
    talib_result = talib.CDLENGULFING(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlgravestonedoji():
    limen_result = cdlgravestonedoji(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlgravestonedoji'].to_numpy()
    talib_result = talib.CDLGRAVESTONEDOJI(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlhammer():
    limen_result = cdlhammer(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlhammer'].to_numpy()
    talib_result = talib.CDLHAMMER(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlhangingman():
    limen_result = cdlhangingman(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlhangingman'].to_numpy()
    talib_result = talib.CDLHANGINGMAN(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlharami():
    limen_result = cdlharami(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlharami'].to_numpy()
    talib_result = talib.CDLHARAMI(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlharamicross():
    limen_result = cdlharamicross(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlharamicross'].to_numpy()
    talib_result = talib.CDLHARAMICROSS(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlhighwave():
    limen_result = cdlhighwave(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlhighwave'].to_numpy()
    talib_result = talib.CDLHIGHWAVE(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlhikkake():
    limen_result = cdlhikkake(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlhikkake'].to_numpy()
    talib_result = talib.CDLHIKKAKE(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlhikkakemod():
    limen_result = cdlhikkakemod(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlhikkakemod'].to_numpy()
    talib_result = talib.CDLHIKKAKEMOD(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlhomingpigeon():
    limen_result = cdlhomingpigeon(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlhomingpigeon'].to_numpy()
    talib_result = talib.CDLHOMINGPIGEON(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlidentical3crows():
    limen_result = cdlidentical3crows(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlidentical3crows'].to_numpy()
    talib_result = talib.CDLIDENTICAL3CROWS(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlinvertedhammer():
    limen_result = cdlinvertedhammer(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlinvertedhammer'].to_numpy()
    talib_result = talib.CDLINVERTEDHAMMER(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlladderbottom():
    limen_result = cdlladderbottom(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlladderbottom'].to_numpy()
    talib_result = talib.CDLLADDERBOTTOM(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdllongleggeddoji():
    limen_result = cdllongleggeddoji(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdllongleggeddoji'].to_numpy()
    talib_result = talib.CDLLONGLEGGEDDOJI(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdllongline():
    limen_result = cdllongline(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdllongline'].to_numpy()
    talib_result = talib.CDLLONGLINE(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlmarubozu():
    limen_result = cdlmarubozu(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlmarubozu'].to_numpy()
    talib_result = talib.CDLMARUBOZU(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlmatchinglow():
    limen_result = cdlmatchinglow(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlmatchinglow'].to_numpy()
    talib_result = talib.CDLMATCHINGLOW(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlmathold():
    limen_result = cdlmathold(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
        penetration=MATHOLD_PENETRATION,
    )['cdlmathold'].to_numpy()
    talib_result = talib.CDLMATHOLD(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        penetration=MATHOLD_PENETRATION,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlonneck():
    limen_result = cdlonneck(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlonneck'].to_numpy()
    talib_result = talib.CDLONNECK(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlpiercing():
    limen_result = cdlpiercing(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlpiercing'].to_numpy()
    talib_result = talib.CDLPIERCING(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlrickshawman():
    limen_result = cdlrickshawman(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlrickshawman'].to_numpy()
    talib_result = talib.CDLRICKSHAWMAN(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlrisefall3methods():
    limen_result = cdlrisefall3methods(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlrisefall3methods'].to_numpy()
    talib_result = talib.CDLRISEFALL3METHODS(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlseparatinglines():
    limen_result = cdlseparatinglines(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlseparatinglines'].to_numpy()
    talib_result = talib.CDLSEPARATINGLINES(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlshootingstar():
    limen_result = cdlshootingstar(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlshootingstar'].to_numpy()
    talib_result = talib.CDLSHOOTINGSTAR(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlshortline():
    limen_result = cdlshortline(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlshortline'].to_numpy()
    talib_result = talib.CDLSHORTLINE(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlspinningtop():
    limen_result = cdlspinningtop(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlspinningtop'].to_numpy()
    talib_result = talib.CDLSPINNINGTOP(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlstalledpattern():
    limen_result = cdlstalledpattern(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlstalledpattern'].to_numpy()
    talib_result = talib.CDLSTALLEDPATTERN(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlsticksandwich():
    limen_result = cdlsticksandwich(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlsticksandwich'].to_numpy()
    talib_result = talib.CDLSTICKSANDWICH(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdltakuri():
    limen_result = cdltakuri(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdltakuri'].to_numpy()
    talib_result = talib.CDLTAKURI(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlthrusting():
    limen_result = cdlthrusting(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlthrusting'].to_numpy()
    talib_result = talib.CDLTHRUSTING(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdltristar():
    limen_result = cdltristar(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdltristar'].to_numpy()
    talib_result = talib.CDLTRISTAR(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cdlunique3river():
    limen_result = cdlunique3river(
        SAMPLE_DATA,
        open_col='open',
        high_col='high',
        low_col='low',
        close_col='close',
    )['cdlunique3river'].to_numpy()
    talib_result = talib.CDLUNIQUE3RIVER(
        NUMPY_DATA['open'],
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
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
    out_col = f"cci_{DEFAULT_PERIOD}"
    limen_result = cci(SAMPLE_DATA, period=DEFAULT_PERIOD)[out_col].to_numpy()
    talib_result = talib.CCI(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        timeperiod=DEFAULT_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_cmo():
    out_col = f"cmo_{DEFAULT_PERIOD}"
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
    out_col = f"mom_{MOM_PERIOD}"
    limen_result = mom(SAMPLE_DATA, price_col='close', period=MOM_PERIOD)[out_col].to_numpy()
    talib_result = talib.MOM(
        NUMPY_DATA['close'],
        timeperiod=MOM_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_roc():
    out_col = f"roc_{ROC_PERIOD}"
    limen_result = roc(SAMPLE_DATA, price_col='close', period=ROC_PERIOD)[out_col].to_numpy()
    talib_result = talib.ROC(
        NUMPY_DATA['close'],
        timeperiod=ROC_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_rocp():
    out_col = f"rocp_{ROCP_PERIOD}"
    limen_result = rocp(SAMPLE_DATA, price_col='close', period=ROCP_PERIOD)[out_col].to_numpy()
    talib_result = talib.ROCP(
        NUMPY_DATA['close'],
        timeperiod=ROCP_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_rocr():
    out_col = f"rocr_{ROCR_PERIOD}"
    limen_result = rocr(SAMPLE_DATA, price_col='close', period=ROCR_PERIOD)[out_col].to_numpy()
    talib_result = talib.ROCR(
        NUMPY_DATA['close'],
        timeperiod=ROCR_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_rocr100():
    out_col = f"rocr100_{ROCR100_PERIOD}"
    limen_result = rocr100(SAMPLE_DATA, price_col='close', period=ROCR100_PERIOD)[out_col].to_numpy()
    talib_result = talib.ROCR100(
        NUMPY_DATA['close'],
        timeperiod=ROCR100_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_rsi():
    out_col = f"rsi_{RSI_PERIOD}"
    limen_result = rsi(SAMPLE_DATA, price_col='close', period=RSI_PERIOD)[out_col].to_numpy()
    talib_result = talib.RSI(
        NUMPY_DATA['close'],
        timeperiod=RSI_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_stoch():
    for slowk_ma_type in MA_TYPES:
        limen = stoch(
            SAMPLE_DATA,
            high_col='high',
            low_col='low',
            close_col='close',
            fastk_period=STOCH_FASTK_PERIOD,
            slowk_period=STOCH_SLOWK_PERIOD,
            slowk_ma_type=slowk_ma_type,
            slowd_period=STOCH_SLOWD_PERIOD,
            slowd_ma_type=0,
        )
        limen_slowk = limen['stoch_slowk'].to_numpy()
        limen_slowd = limen['stoch_slowd'].to_numpy()

        talib_slowk, talib_slowd = talib.STOCH(
            NUMPY_DATA['high'],
            NUMPY_DATA['low'],
            NUMPY_DATA['close'],
            fastk_period=STOCH_FASTK_PERIOD,
            slowk_period=STOCH_SLOWK_PERIOD,
            slowk_matype=slowk_ma_type,
            slowd_period=STOCH_SLOWD_PERIOD,
            slowd_matype=0,
        )

        verify_indicator_results_match(limen_slowk, talib_slowk)
        verify_indicator_results_match(limen_slowd, talib_slowd)

    for slowd_ma_type in MA_TYPES:
        limen = stoch(
            SAMPLE_DATA,
            high_col='high',
            low_col='low',
            close_col='close',
            fastk_period=STOCH_FASTK_PERIOD,
            slowk_period=STOCH_SLOWK_PERIOD,
            slowk_ma_type=0,
            slowd_period=STOCH_SLOWD_PERIOD,
            slowd_ma_type=slowd_ma_type,
        )
        limen_slowk = limen['stoch_slowk'].to_numpy()
        limen_slowd = limen['stoch_slowd'].to_numpy()

        talib_slowk, talib_slowd = talib.STOCH(
            NUMPY_DATA['high'],
            NUMPY_DATA['low'],
            NUMPY_DATA['close'],
            fastk_period=STOCH_FASTK_PERIOD,
            slowk_period=STOCH_SLOWK_PERIOD,
            slowk_matype=0,
            slowd_period=STOCH_SLOWD_PERIOD,
            slowd_matype=slowd_ma_type,
        )

        verify_indicator_results_match(limen_slowk, talib_slowk)
        verify_indicator_results_match(limen_slowd, talib_slowd)


def test_stochf():
    for fastd_ma_type in MA_TYPES:
        limen = stochf(
            SAMPLE_DATA,
            high_col='high',
            low_col='low',
            close_col='close',
            fastk_period=STOCHF_FASTK_PERIOD,
            fastd_period=STOCHF_FASTD_PERIOD,
            fastd_ma_type=fastd_ma_type,
        )
        limen_fastk = limen['stochf_fastk'].to_numpy()
        limen_fastd = limen['stochf_fastd'].to_numpy()

        talib_fastk, talib_fastd = talib.STOCHF(
            NUMPY_DATA['high'],
            NUMPY_DATA['low'],
            NUMPY_DATA['close'],
            fastk_period=STOCHF_FASTK_PERIOD,
            fastd_period=STOCHF_FASTD_PERIOD,
            fastd_matype=fastd_ma_type,
        )

        verify_indicator_results_match(limen_fastk, talib_fastk)
        verify_indicator_results_match(limen_fastd, talib_fastd)


def test_stochrsi():
    for fastd_ma_type in MA_TYPES:
        limen = stochrsi(
            SAMPLE_DATA,
            price_col='close',
            period=STOCHRSI_PERIOD,
            fastk_period=STOCHRSI_FASTK_PERIOD,
            fastd_period=STOCHRSI_FASTD_PERIOD,
            fastd_ma_type=fastd_ma_type,
        )
        limen_fastk = limen['stochrsi_fastk'].to_numpy()
        limen_fastd = limen['stochrsi_fastd'].to_numpy()

        talib_fastk, talib_fastd = talib.STOCHRSI(
            NUMPY_DATA['close'],
            timeperiod=STOCHRSI_PERIOD,
            fastk_period=STOCHRSI_FASTK_PERIOD,
            fastd_period=STOCHRSI_FASTD_PERIOD,
            fastd_matype=fastd_ma_type,
        )

        verify_indicator_results_match(limen_fastk, talib_fastk)
        verify_indicator_results_match(limen_fastd, talib_fastd)


def test_ultosc():
    out_col = f"ultosc_{ULTOSC_PERIOD1}_{ULTOSC_PERIOD2}_{ULTOSC_PERIOD3}"
    limen_result = ultosc(
        SAMPLE_DATA,
        high_col='high',
        low_col='low',
        close_col='close',
        period1=ULTOSC_PERIOD1,
        period2=ULTOSC_PERIOD2,
        period3=ULTOSC_PERIOD3,
    )[out_col].to_numpy()
    talib_result = talib.ULTOSC(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        timeperiod1=ULTOSC_PERIOD1,
        timeperiod2=ULTOSC_PERIOD2,
        timeperiod3=ULTOSC_PERIOD3,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_willr():
    out_col = f"willr_{WILLR_PERIOD}"
    limen_result = willr(
        SAMPLE_DATA,
        high_col='high',
        low_col='low',
        close_col='close',
        period=WILLR_PERIOD,
    )[out_col].to_numpy()
    talib_result = talib.WILLR(
        NUMPY_DATA['high'],
        NUMPY_DATA['low'],
        NUMPY_DATA['close'],
        timeperiod=WILLR_PERIOD,
    )

    verify_indicator_results_match(limen_result, talib_result)


def test_natr():
    out_col = f"natr_{DEFAULT_PERIOD}"
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
    max_diff = float(np.max(diff))  # float for cleaner printing

    assert max_diff < tolerance, f"Results differ, Max Diff: {max_diff}"


def test_indicators_vs_talib():
    test_functions = [
        test_ad,
        test_adosc,
        test_apo,
        test_ppo,
        test_dema,
        test_ema,
        test_ht_trendline,
        test_ht_dcperiod,
        test_ht_dcphase,
        test_ht_phasor,
        test_ht_sine,
        test_ht_trendmode,
        test_kama,
        test_ma,
        test_mama,
        test_sma,
        test_tema,
        test_trima,
        test_trix,
        test_t3,
        test_tsf,
        test_linearreg,
        test_linearreg_angle,
        test_linearreg_intercept,
        test_linearreg_slope,
        test_stddev,
        test_var,
        test_wma,
        test_sar,
        test_sarext,
        test_mfi,
        test_atr,
        test_avgprice,
        test_medprice,
        test_midpoint,
        test_midprice,
        test_typprice,
        test_wclprice,
        test_cdl2crows,
        test_cdl3blackcrows,
        test_cdl3inside,
        test_cdl3linestrike,
        test_cdl3starsinsouth,
        test_cdl3whitesoldiers,
        test_cdlabandonedbaby,
        test_cdladvancedblock,
        test_cdlbelthold,
        test_cdlclosingmarubozu,
        test_cdlconcealbabyswall,
        test_cdlcounterattack,
        test_cdldarkcloudcover,
        test_coldoji,
        test_cdldragonflydoji,
        test_cdlengulfing,
        test_cdlgravestonedoji,
        test_cdlhammer,
        test_cdlhangingman,
        test_cdlharami,
        test_cdlharamicross,
        test_cdlhighwave,
        test_cdlhikkake,
        test_cdlhikkakemod,
        test_cdlhomingpigeon,
        test_cdlidentical3crows,
        test_cdlinvertedhammer,
        test_cdlladderbottom,
        test_cdllongleggeddoji,
        test_cdllongline,
        test_cdlmarubozu,
        test_cdlmatchinglow,
        test_cdlmathold,
        test_cdlonneck,
        test_cdlpiercing,
        test_cdlrickshawman,
        test_cdlrisefall3methods,
        test_cdlseparatinglines,
        test_cdlshootingstar,
        test_cdlshortline,
        test_cdlspinningtop,
        test_cdlstalledpattern,
        test_cdlsticksandwich,
        test_cdltakuri,
        test_cdlthrusting,
        test_cdltristar,
        test_cdlunique3river,
        test_bbands,
        test_bop,
        test_cci,
        test_cmo,
        test_macd,
        test_macdfix,
        test_macdext,
        test_mom,
        test_roc,
        test_rocp,
        test_rocr,
        test_rocr100,
        test_rsi,
        test_stoch,
        test_stochf,
        test_stochrsi,
        test_ultosc,
        test_willr,
        test_natr,
        test_obv,
        test_trange,
    ]

    for test_func in test_functions:
        test_func()
