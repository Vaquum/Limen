# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Vaquum/Limen/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                    |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------------------- | -------: | -------: | ------: | --------: |
| limen/\_\_init\_\_.py                                   |       19 |        0 |    100% |           |
| limen/backtest/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| limen/backtest/backtest\_sequential.py                  |       70 |        3 |     96% |33, 41, 104 |
| limen/backtest/backtest\_snapshot.py                    |       46 |        2 |     96% |     58-59 |
| limen/cohort/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| limen/cohort/regime\_pools.py                           |      219 |      103 |     53% |46, 56-74, 87-102, 118-174, 211-213, 252, 258-263, 267-282, 288-292, 399-428, 467 |
| limen/data/\_\_init\_\_.py                              |        3 |        0 |    100% |           |
| limen/data/\_internal/binance\_file\_to\_polars.py      |       13 |        0 |    100% |           |
| limen/data/bars/\_\_init\_\_.py                         |        2 |        0 |    100% |           |
| limen/data/bars/standard\_bars.py                       |       28 |        0 |    100% |           |
| limen/data/historical\_data.py                          |      195 |       60 |     69% |40-47, 58, 61, 71, 80-82, 90-97, 109-124, 137-150, 157-161, 178-183, 190, 197, 202, 210, 220, 226, 264, 277, 314, 323, 329, 355, 465-472, 495-498 |
| limen/data/utils/\_\_init\_\_.py                        |        6 |        0 |    100% |           |
| limen/data/utils/compute\_data\_bars.py                 |       20 |       16 |     20% |     20-40 |
| limen/data/utils/random\_slice.py                       |       13 |        2 |     85% |    26, 33 |
| limen/data/utils/splits.py                              |       45 |        7 |     84% |22, 59-64, 97 |
| limen/experiment/\_\_init\_\_.py                        |        9 |        0 |    100% |           |
| limen/experiment/checkpoint\_manager.py                 |       92 |        6 |     93% |240, 266, 292-295 |
| limen/experiment/experiment\_core.py                    |      347 |       43 |     88% |68, 79, 93-102, 156, 171, 175, 183, 186, 201, 205, 217, 224-225, 229-230, 252, 417, 419, 422, 502, 506, 609, 617, 633, 640, 669, 672-673, 714, 753, 773-774, 866, 872, 875-876, 945-946 |
| limen/experiment/feedback\_controller.py                |      131 |        9 |     93% |36-37, 42, 48-49, 97, 291, 363-364 |
| limen/experiment/manifest\_core.py                      |      385 |       39 |     90% |81-84, 100, 107, 245-248, 255, 272, 346-348, 406-408, 432, 526-527, 554, 556, 558, 564, 596-598, 679, 695, 731, 745, 810, 865-868, 888, 922, 958, 1103, 1120 |
| limen/experiment/msq.py                                 |      193 |        7 |     96% |197, 210, 334, 359, 462-463, 475 |
| limen/experiment/param\_domain.py                       |      129 |        3 |     98% |87, 183, 244 |
| limen/experiment/param\_search/\_\_init\_\_.py          |        5 |        0 |    100% |           |
| limen/experiment/param\_search/grid\_strategy.py        |       70 |        2 |     97% |  119, 124 |
| limen/experiment/param\_search/random\_strategy.py      |       32 |        0 |    100% |           |
| limen/experiment/param\_search/registry.py              |        5 |        0 |    100% |           |
| limen/experiment/param\_search/search\_strategy.py      |       52 |        1 |     98% |        65 |
| limen/experiment/reducer/\_\_init\_\_.py                |       15 |        0 |    100% |           |
| limen/experiment/reducer/budget\_reducer.py             |      120 |        8 |     93% |127, 179, 227, 233, 243, 249, 258, 262 |
| limen/experiment/reducer/correlation\_reducer.py        |      119 |       12 |     90% |73, 78, 83, 88, 93, 98, 147, 157, 167, 257, 262, 292 |
| limen/experiment/reducer/filter\_types.py               |       23 |        5 |     78% |     45-51 |
| limen/experiment/reducer/focus\_reducer.py              |      162 |       13 |     92% |63, 68, 73, 78, 123, 143, 227, 257, 301, 310, 321, 403, 407 |
| limen/experiment/reducer/pruning\_strategy.py           |       19 |        0 |    100% |           |
| limen/experiment/reducer/registry.py                    |        7 |        0 |    100% |           |
| limen/experiment/reducer/sanity\_reducer.py             |       81 |        8 |     90% |59, 64, 69, 74, 79, 154, 192, 226 |
| limen/experiment/reducer/saturation\_reducer.py         |       62 |        2 |     97% |   48, 103 |
| limen/experiment/trainer/\_\_init\_\_.py                |        4 |        0 |    100% |           |
| limen/experiment/trainer/errors.py                      |        1 |        0 |    100% |           |
| limen/experiment/trainer/sensor.py                      |       29 |        1 |     97% |        81 |
| limen/experiment/trainer/trainer.py                     |      129 |       23 |     82% |60-61, 67, 75, 89, 107-108, 117, 120-125, 144-149, 171, 185, 191, 218, 222, 229, 234, 236, 247 |
| limen/features/\_\_init\_\_.py                          |       31 |        0 |    100% |           |
| limen/features/active\_lines.py                         |       25 |       25 |      0% |      1-45 |
| limen/features/active\_quantile\_count.py               |       25 |       25 |      0% |      1-44 |
| limen/features/atr\_percent\_sma.py                     |        3 |        1 |     67% |        17 |
| limen/features/atr\_sma.py                              |        3 |        1 |     67% |        19 |
| limen/features/breakout\_features.py                    |       40 |       34 |     15% |25-34, 55, 94-103, 137-161 |
| limen/features/breakout\_percentile\_regime.py          |        6 |        3 |     50% |     24-27 |
| limen/features/close\_position.py                       |        3 |        1 |     67% |        16 |
| limen/features/close\_to\_extremes.py                   |        3 |        3 |      0% |      1-16 |
| limen/features/conserved\_flux\_renormalization.py      |       34 |        2 |     94% |   66, 118 |
| limen/features/distance\_from\_high.py                  |        3 |        1 |     67% |        17 |
| limen/features/distance\_from\_low.py                   |        3 |        1 |     67% |        17 |
| limen/features/dynamic\_stop\_loss.py                   |        3 |        3 |      0% |      1-24 |
| limen/features/dynamic\_target.py                       |        3 |        3 |      0% |      1-24 |
| limen/features/ema\_alignment.py                        |        4 |        4 |      0% |      1-22 |
| limen/features/ema\_breakout.py                         |        5 |        3 |     40% |     24-31 |
| limen/features/entry\_score\_microstructure.py          |       13 |       13 |      0% |      1-94 |
| limen/features/exit\_quality.py                         |        3 |        3 |      0% |      1-23 |
| limen/features/feature\_aliases.py                      |        6 |        6 |      0% |      1-36 |
| limen/features/forward\_breakout\_target.py             |        9 |        7 |     22% |     25-37 |
| limen/features/fractional\_diff.py                      |       70 |        6 |     91% |29, 79, 100-101, 181-182 |
| limen/features/gap\_high.py                             |        3 |        1 |     67% |        16 |
| limen/features/hh\_hl\_structure\_regime.py             |        8 |        6 |     25% |     20-27 |
| limen/features/hours\_since\_big\_move.py               |       18 |       18 |      0% |      1-43 |
| limen/features/hours\_since\_quantile\_line.py          |       18 |       18 |      0% |      1-43 |
| limen/features/ichimoku\_cloud.py                       |        3 |        1 |     67% |        27 |
| limen/features/kline\_imbalance.py                      |        3 |        0 |    100% |           |
| limen/features/lagged\_features.py                      |       21 |        9 |     57% |23, 26, 29, 32, 62, 102-107 |
| limen/features/log\_returns.py                          |        3 |        3 |      0% |      1-17 |
| limen/features/ma\_slope\_regime.py                     |        7 |        5 |     29% |     22-28 |
| limen/features/market\_regime.py                        |       21 |       21 |      0% |      1-58 |
| limen/features/micro\_momentum.py                       |        3 |        3 |      0% |      1-17 |
| limen/features/momentum\_confirmation.py                |        7 |        7 |      0% |      1-32 |
| limen/features/momentum\_periods.py                     |        9 |        9 |      0% |      1-28 |
| limen/features/momentum\_weight.py                      |        3 |        3 |      0% |      1-19 |
| limen/features/position\_in\_candle.py                  |        4 |        4 |      0% |      1-19 |
| limen/features/position\_in\_range.py                   |        3 |        3 |      0% |      1-18 |
| limen/features/price\_range\_position.py                |        4 |        1 |     75% |        19 |
| limen/features/price\_vs\_band\_regime.py               |       11 |        8 |     27% |     23-34 |
| limen/features/quantile\_flag.py                        |        5 |        0 |    100% |           |
| limen/features/quantile\_line\_density.py               |       19 |       19 |      0% |      1-40 |
| limen/features/range\_pct.py                            |        3 |        0 |    100% |           |
| limen/features/regime\_multiplier.py                    |        3 |        3 |      0% |      1-23 |
| limen/features/returns\_lags.py                         |        6 |        6 |      0% |      1-26 |
| limen/features/risk\_reward\_ratio.py                   |        4 |        4 |      0% |      1-19 |
| limen/features/sma\_crossover.py                        |        3 |        1 |     67% |        26 |
| limen/features/sma\_ratios.py                           |       13 |        1 |     92% |        22 |
| limen/features/spread.py                                |        3 |        3 |      0% |      1-15 |
| limen/features/spread\_percent.py                       |        3 |        3 |      0% |      1-16 |
| limen/features/time\_features.py                        |        3 |        3 |      0% |      1-16 |
| limen/features/trend\_strength.py                       |        3 |        1 |     67% |        18 |
| limen/features/volatility\_1h.py                        |        3 |        3 |      0% |      1-16 |
| limen/features/volatility\_measure.py                   |        3 |        3 |      0% |      1-16 |
| limen/features/volatility\_weight.py                    |        8 |        8 |      0% |      1-37 |
| limen/features/volume\_ratio.py                         |        7 |        1 |     86% |        21 |
| limen/features/volume\_regime.py                        |        3 |        1 |     67% |        17 |
| limen/features/volume\_spike.py                         |        5 |        5 |      0% |      1-24 |
| limen/features/volume\_trend.py                         |       10 |       10 |      0% |      1-29 |
| limen/features/volume\_weight.py                        |        6 |        6 |      0% |      1-27 |
| limen/features/vwap.py                                  |        3 |        0 |    100% |           |
| limen/features/window\_return\_regime.py                |        6 |        3 |     50% |     23-26 |
| limen/indicators/\_\_init\_\_.py                        |      118 |        0 |    100% |           |
| limen/indicators/\_atr.py                               |       30 |        7 |     77% |12-14, 17, 23, 32-33 |
| limen/indicators/\_bbands.py                            |       54 |       54 |      0% |      1-85 |
| limen/indicators/\_ema.py                               |       21 |        3 |     86% | 18, 26-27 |
| limen/indicators/\_hilbert.py                           |       21 |        1 |     95% |        25 |
| limen/indicators/ad.py                                  |        5 |        0 |    100% |           |
| limen/indicators/adosc.py                               |       15 |        2 |     87% |    34, 36 |
| limen/indicators/apo.py                                 |       21 |        4 |     81% |33, 35, 37, 39 |
| limen/indicators/atr.py                                 |       14 |        1 |     93% |        34 |
| limen/indicators/avgprice.py                            |        4 |        0 |    100% |           |
| limen/indicators/bbands.py                              |       43 |       14 |     67% |61, 63, 65, 67, 86-87, 92-100 |
| limen/indicators/body\_pct.py                           |        3 |        0 |    100% |           |
| limen/indicators/bollinger\_bands.py                    |        6 |        4 |     33% |     24-28 |
| limen/indicators/bollinger\_position.py                 |        6 |        3 |     50% |     20-28 |
| limen/indicators/bop.py                                 |        6 |        0 |    100% |           |
| limen/indicators/cci.py                                 |       33 |        3 |     91% |20, 38, 66 |
| limen/indicators/cdl2crows.py                           |       44 |        2 |     95% |    41, 84 |
| limen/indicators/cdl3blackcrows.py                      |       43 |        2 |     95% |    43, 81 |
| limen/indicators/cdl3inside.py                          |       53 |        1 |     98% |        44 |
| limen/indicators/cdl3linestrike.py                      |       42 |        1 |     98% |        43 |
| limen/indicators/cdl3starsinsouth.py                    |       83 |        5 |     94% |53, 72-73, 105, 125 |
| limen/indicators/cdl3whitesoldiers.py                   |       89 |        2 |     98% |   55, 127 |
| limen/indicators/cdlabandonedbaby.py                    |       72 |        3 |     96% |40, 58, 122 |
| limen/indicators/cdladvancedblock.py                    |      116 |        5 |     96% |55, 84-86, 131 |
| limen/indicators/cdlbelthold.py                         |       52 |        1 |     98% |        47 |
| limen/indicators/cdlclosingmarubozu.py                  |       52 |        1 |     98% |        47 |
| limen/indicators/cdlconcealbabyswall.py                 |       50 |        2 |     96% |    43, 89 |
| limen/indicators/cdlcounterattack.py                    |       54 |        2 |     96% |    46, 82 |
| limen/indicators/cdldarkcloudcover.py                   |       40 |        3 |     92% |36, 49, 75 |
| limen/indicators/cdldoji.py                             |       37 |        1 |     97% |        44 |
| limen/indicators/cdldragonflydoji.py                    |       53 |        1 |     98% |        49 |
| limen/indicators/cdlengulfing.py                        |       27 |        1 |     96% |        37 |
| limen/indicators/cdlgravestonedoji.py                   |       53 |        1 |     98% |        49 |
| limen/indicators/cdlhammer.py                           |       81 |        4 |     95% |58, 77-78, 103 |
| limen/indicators/cdlhangingman.py                       |       81 |        4 |     95% |58, 77-78, 103 |
| limen/indicators/cdlharami.py                           |       55 |        1 |     98% |        44 |
| limen/indicators/cdlharamicross.py                      |       58 |        1 |     98% |        47 |
| limen/indicators/cdlhighwave.py                         |       54 |        4 |     93% |47, 61-62, 74 |
| limen/indicators/cdlhikkake.py                          |       38 |        4 |     89% |38, 54-55, 63 |
| limen/indicators/cdlhikkakemod.py                       |       58 |        4 |     93% |44, 78-79, 87 |
| limen/indicators/cdlhomingpigeon.py                     |       49 |        2 |     96% |    44, 78 |
| limen/indicators/cdlidentical3crows.py                  |       62 |        2 |     97% |    47, 97 |
| limen/indicators/cdlinvertedhammer.py                   |       68 |        4 |     94% |52, 68-69, 87 |
| limen/indicators/cdlladderbottom.py                     |       42 |        2 |     95% |    44, 79 |
| limen/indicators/cdllongleggeddoji.py                   |       55 |        4 |     93% |49, 63-64, 76 |
| limen/indicators/cdllongline.py                         |       52 |        1 |     98% |        45 |
| limen/indicators/cdlmarubozu.py                         |       52 |        1 |     98% |        47 |
| limen/indicators/cdlmatchinglow.py                      |       38 |        1 |     97% |        44 |
| limen/indicators/cdlmathold.py                          |       64 |        3 |     95% |36, 50, 111 |
| limen/indicators/cdlonneck.py                           |       51 |        2 |     96% |    47, 80 |
| limen/indicators/cdlpiercing.py                         |       40 |        2 |     95% |    41, 70 |
| limen/indicators/cdlrickshawman.py                      |       73 |        4 |     95% |54, 70-71, 93 |
| limen/indicators/cdlrisefall3methods.py                 |       61 |        1 |     98% |        43 |
| limen/indicators/cdlseparatinglines.py                  |       67 |        1 |     99% |        52 |
| limen/indicators/cdlshootingstar.py                     |       68 |        4 |     94% |52, 68-69, 87 |
| limen/indicators/cdlshortline.py                        |       52 |        1 |     98% |        45 |
| limen/indicators/cdlspinningtop.py                      |       38 |        1 |     97% |        42 |
| limen/indicators/cdlstalledpattern.py                   |       81 |        2 |     98% |   54, 116 |
| limen/indicators/cdlsticksandwich.py                    |       39 |        2 |     95% |    44, 69 |
| limen/indicators/cdltakuri.py                           |       69 |        4 |     94% |54, 75-76, 89 |
| limen/indicators/cdlthrusting.py                        |       51 |        2 |     96% |    47, 80 |
| limen/indicators/cdltristar.py                          |       44 |        3 |     93% |44, 73, 78 |
| limen/indicators/cdlunique3river.py                     |       51 |        2 |     96% |    45, 84 |
| limen/indicators/cmo.py                                 |       60 |        4 |     93% |17, 46, 70, 95 |
| limen/indicators/dema.py                                |       31 |        4 |     87% |17, 24, 28, 55 |
| limen/indicators/ema.py                                 |       23 |        2 |     91% |    15, 43 |
| limen/indicators/ht\_dcperiod.py                        |      110 |        1 |     99% |        21 |
| limen/indicators/ht\_dcphase.py                         |      149 |        3 |     98% |24, 217, 219 |
| limen/indicators/ht\_phasor.py                          |      113 |      102 |     10% |    15-198 |
| limen/indicators/ht\_sine.py                            |      152 |      137 |     10% |    21-243 |
| limen/indicators/ht\_trendline.py                       |      135 |        1 |     99% |        24 |
| limen/indicators/ht\_trendmode.py                       |      191 |        3 |     98% |28, 233, 235 |
| limen/indicators/kama.py                                |       79 |       16 |     80% |17, 55-72, 120 |
| limen/indicators/linearreg.py                           |       35 |        2 |     94% |    14, 62 |
| limen/indicators/linearreg\_angle.py                    |       36 |        2 |     94% |    15, 63 |
| limen/indicators/linearreg\_intercept.py                |       34 |        2 |     94% |    14, 61 |
| limen/indicators/linearreg\_slope.py                    |       33 |        2 |     94% |    14, 60 |
| limen/indicators/ma.py                                  |       46 |        3 |     93% |44, 46, 51 |
| limen/indicators/macd.py                                |       44 |        5 |     89% |28, 33, 77, 79, 81 |
| limen/indicators/macdext.py                             |       73 |       12 |     84% |17, 31, 44, 79-80, 89, 156, 158, 160, 162, 164, 166 |
| limen/indicators/macdfix.py                             |       41 |        2 |     95% |    32, 84 |
| limen/indicators/mama.py                                |      135 |        3 |     98% |27, 248, 250 |
| limen/indicators/medprice.py                            |        4 |        0 |    100% |           |
| limen/indicators/mfi.py                                 |       12 |        1 |     92% |        32 |
| limen/indicators/midpoint.py                            |       32 |        2 |     94% |    14, 56 |
| limen/indicators/midprice.py                            |       31 |        2 |     94% |    14, 57 |
| limen/indicators/mom.py                                 |        8 |        1 |     88% |        25 |
| limen/indicators/natr.py                                |       19 |        2 |     89% |    36, 50 |
| limen/indicators/obv.py                                 |        4 |        0 |    100% |           |
| limen/indicators/ppo.py                                 |       25 |        4 |     84% |34, 36, 38, 40 |
| limen/indicators/price\_change\_pct.py                  |        3 |        1 |     67% |        18 |
| limen/indicators/returns.py                             |        3 |        1 |     67% |        16 |
| limen/indicators/roc.py                                 |        9 |        1 |     89% |        25 |
| limen/indicators/rocp.py                                |        9 |        1 |     89% |        25 |
| limen/indicators/rocr100.py                             |        9 |        1 |     89% |        25 |
| limen/indicators/rocr.py                                |        9 |        1 |     89% |        25 |
| limen/indicators/rolling\_volatility.py                 |        3 |        0 |    100% |           |
| limen/indicators/rsi.py                                 |       43 |        2 |     95% |    15, 75 |
| limen/indicators/rsi\_sma.py                            |        3 |        1 |     67% |        20 |
| limen/indicators/sar.py                                 |       85 |        6 |     93% |16, 23, 38-39, 140, 142 |
| limen/indicators/sarext.py                              |      119 |       25 |     79% |23, 32-33, 37-38, 45-48, 60-67, 89, 121, 184, 186, 188, 190, 192, 194, 196, 198 |
| limen/indicators/sma.py                                 |       35 |        2 |     94% |    14, 59 |
| limen/indicators/sma\_deviation\_std.py                 |        6 |        4 |     33% |     20-24 |
| limen/indicators/stddev.py                              |       54 |        5 |     91% |20, 54, 56, 85, 87 |
| limen/indicators/stoch.py                               |      106 |        9 |     92% |16, 30, 55, 106, 174, 176, 178, 180, 182 |
| limen/indicators/stochastic\_oscillator.py              |        7 |        0 |    100% |           |
| limen/indicators/stochf.py                              |       97 |        7 |     93% |16, 30, 52, 103, 156, 158, 160 |
| limen/indicators/stochrsi.py                            |       33 |        5 |     85% |26, 69, 71, 73, 75 |
| limen/indicators/t3.py                                  |      109 |       10 |     91% |14, 88-94, 141, 143 |
| limen/indicators/tema.py                                |       31 |        5 |     84% |17, 21, 25, 29, 55 |
| limen/indicators/trange.py                              |        6 |        0 |    100% |           |
| limen/indicators/trima.py                               |      107 |        2 |     98% |   14, 146 |
| limen/indicators/trix.py                                |       35 |        3 |     91% |16, 37, 62 |
| limen/indicators/tsf.py                                 |       37 |        2 |     95% |    14, 62 |
| limen/indicators/typprice.py                            |        4 |        0 |    100% |           |
| limen/indicators/ultosc.py                              |       88 |        4 |     95% |48, 142, 144, 146 |
| limen/indicators/var.py                                 |       47 |        3 |     94% |18, 77, 79 |
| limen/indicators/wclprice.py                            |        4 |        0 |    100% |           |
| limen/indicators/wilder\_rsi.py                         |        3 |        0 |    100% |           |
| limen/indicators/willr.py                               |       13 |        1 |     92% |        32 |
| limen/indicators/window\_return.py                      |        4 |        0 |    100% |           |
| limen/indicators/wma.py                                 |       47 |        2 |     96% |    17, 78 |
| limen/log/\_\_init\_\_.py                               |        6 |        0 |    100% |           |
| limen/log/\_experiment\_backtest\_results.py            |       11 |        0 |    100% |           |
| limen/log/\_experiment\_confusion\_metrics.py           |       10 |        0 |    100% |           |
| limen/log/\_experiment\_parameter\_correlation.py       |       83 |       12 |     86% |48-49, 64, 67, 77, 91, 100, 111-112, 117-118, 151 |
| limen/log/\_permutation\_confusion\_metrics.py          |       74 |        9 |     88% |44-47, 52, 61-65, 69, 116 |
| limen/log/\_permutation\_prediction\_performance.py     |       19 |        3 |     84% | 20-21, 24 |
| limen/log/\_read\_from\_file.py                         |       16 |        1 |     94% |        35 |
| limen/log/log.py                                        |       47 |        4 |     91% |48, 54, 66, 92 |
| limen/metrics/\_\_init\_\_.py                           |        6 |        0 |    100% |           |
| limen/metrics/balanced\_metric.py                       |        9 |        5 |     44% |     23-29 |
| limen/metrics/binary\_metrics.py                        |        4 |        0 |    100% |           |
| limen/metrics/continuous\_metrics.py                    |       12 |        0 |    100% |           |
| limen/metrics/multiclass\_metrics.py                    |        5 |        2 |     60% |     21-26 |
| limen/metrics/safe\_ovr\_auc.py                         |       12 |        9 |     25% |     17-27 |
| limen/scalers/\_\_init\_\_.py                           |        6 |        0 |    100% |           |
| limen/scalers/linear\_scaler.py                         |       58 |       20 |     66% |68-75, 96, 159, 180-196 |
| limen/scalers/logreg\_scaler.py                         |       46 |       15 |     67% |91, 112-133 |
| limen/scalers/rank\_gauss\_scaler.py                    |       51 |        4 |     92% |39, 69, 109, 122 |
| limen/scalers/registry.py                               |        5 |        0 |    100% |           |
| limen/scalers/robust\_scaler.py                         |       35 |        2 |     94% |    68, 94 |
| limen/sfd/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| limen/sfd/foundational\_sfd/\_\_init\_\_.py             |        8 |        0 |    100% |           |
| limen/sfd/foundational\_sfd/logreg\_binary.py           |       17 |        0 |    100% |           |
| limen/sfd/foundational\_sfd/random\_binary.py           |        9 |        0 |    100% |           |
| limen/sfd/foundational\_sfd/tabpfn\_binary.py           |       12 |        7 |     42% |     17-44 |
| limen/sfd/foundational\_sfd/xgboost\_regressor.py       |       17 |        0 |    100% |           |
| limen/sfd/reference\_architecture/\_\_init\_\_.py       |       14 |        1 |     93% |        12 |
| limen/sfd/reference\_architecture/base.py               |       40 |        2 |     95% |  126, 138 |
| limen/sfd/reference\_architecture/logreg\_binary.py     |       30 |        0 |    100% |           |
| limen/sfd/reference\_architecture/random\_binary.py     |       30 |        0 |    100% |           |
| limen/sfd/reference\_architecture/tabpfn\_binary.py     |       55 |       52 |      5% |     7-174 |
| limen/sfd/reference\_architecture/xgboost\_regressor.py |       34 |        0 |    100% |           |
| limen/trading/\_\_init\_\_.py                           |        2 |        0 |    100% |           |
| limen/trading/account.py                                |       99 |       10 |     90% |69, 71, 73, 83, 85, 87, 92, 120, 159, 161 |
| limen/transforms/\_\_init\_\_.py                        |        8 |        0 |    100% |           |
| limen/transforms/calibrate\_classifier.py               |        7 |        3 |     57% |     26-29 |
| limen/transforms/mad\_transform.py                      |       10 |        8 |     20% |     17-34 |
| limen/transforms/optimize\_binary\_threshold.py         |       17 |       13 |     24% |     31-51 |
| limen/transforms/quantile\_trim\_transform.py           |       15 |       13 |     13% |     17-41 |
| limen/transforms/shift\_column\_transform.py            |        3 |        0 |    100% |           |
| limen/transforms/winsorize\_transform.py                |       14 |       12 |     14% |     18-42 |
| limen/transforms/zscore\_transform.py                   |       11 |        9 |     18% |     17-38 |
| limen/utils/\_\_init\_\_.py                             |       10 |        0 |    100% |           |
| limen/utils/adf\_test.py                                |       18 |        1 |     94% |        42 |
| limen/utils/confidence\_filtering\_system.py            |       55 |        3 |     95% |48, 116, 122 |
| limen/utils/data\_dict\_to\_numpy.py                    |        9 |        8 |     11% |     14-23 |
| limen/utils/param\_space.py                             |       31 |        1 |     97% |        56 |
| limen/utils/reporting.py                                |        8 |        5 |     38% |14-15, 30-31, 45 |
| **TOTAL**                                               | **10446** | **1583** | **85%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Vaquum/Limen/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Vaquum/Limen/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Vaquum/Limen/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Vaquum/Limen/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FVaquum%2FLimen%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Vaquum/Limen/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.