# Benchmark

In Limen, benchmark is the prediction-quality layer between the raw experiment log and the trading backtest.

It measures signal activity, positive-class accuracy, and true-positive versus false-positive outcome separation. Benchmark comes before PnL compression so prediction quality remains visible before a trading rule is applied.

## Prerequisites

- a completed experiment log with predictions and targets
- retained round artifacts (`post_processing=True`) for direct UEL analysis
- a numeric outcome column such as `price_change` for return-separation fields

## Risk boundary

Benchmark output is research evidence, not investment advice, trading advice, regulatory approval, or a promise of future performance. Past performance is not predictive, digital-asset trading can result in total loss of capital, and a benchmark table can show statistical structure without proving that a strategy survives live execution, fees, slippage, or portfolio constraints.

## Where benchmark lives

Benchmark analytics are built on top of `Log`.

The confusion tables diagnose prediction quality. Walk-forward sweeps additionally retain test return tracks and produce the selection report described below. The report assesses the evaluated sweep; it provides no leaderboard or independent evidence of future performance.

The main surfaces are:

- `uel.experiment_confusion_metrics`
- `uel._log.experiment_confusion_metrics('price_change')`
- `uel._log.permutation_confusion_metrics(x='price_change', round_id=0)`

The first two are the same analysis surfaced in two places: UEL computes the experiment-wide confusion table automatically at the end of a run.

## Benchmark workflow

```python
benchmark = uel.experiment_confusion_metrics

round0 = uel._log.permutation_confusion_metrics(
    x='price_change',
    round_id=0,
)
```

Use the experiment-wide table to compare rounds. Use the single-round table when one permutation deserves a closer look.

## What benchmark measures

The current benchmark table focuses on long-only prediction quality and outcome separation.

Key fields include:

- `pred_pos_rate_pct`
- `actual_pos_rate_pct`
- `precision_pct`
- `recall_pct`
- `pred_pos_count`
- `tp_count`
- `fp_count`
- `tp_x_mean`, `tp_x_median`
- `fp_x_mean`, `fp_x_median`
- `tp_fp_cohen_d`
- `tp_fp_ks`

When `x='price_change'`, the table reports the long-call rate, realized long-class accuracy, and realized price-change separation between true positives and false positives.

## How to read it

### Signal rate

`pred_pos_rate_pct` reports signal activity. High precision with a low positive rate can leave too few candidate bars for downstream use.

### Precision and recall

- `precision_pct` is the share of predicted positives that were true positives.
- `recall_pct` is the share of actual positives captured by the model.

Interpret precision and recall together. High precision with low recall means a selective signal. High recall with low precision means a noisy signal.

### TP versus FP quality

Limen's benchmark layer extends precision and recall with outcome-separation metrics.

- `tp_x_mean` and `tp_x_median` describe the chosen outcome inside true positives
- `fp_x_mean` and `fp_x_median` do the same for false positives
- `tp_fp_cohen_d` and `tp_fp_ks` estimate how separated those two distributions are

If TP and FP are not separated on the chosen outcome, a round can be statistically correct while adding little downstream signal value.

## Benchmark versus backtest

Benchmark and backtest are intentionally separate:

- benchmark asks whether the signal has predictive structure
- backtest asks whether that structure survives a concrete long-only trading rule with costs

This separation matters because the layers can disagree.

Common cases:

- a round can have high benchmark metrics but low trading economics
- a round can have modest benchmark metrics yet produce usable backtest behavior because of position profile and cost structure

Limen exposes both layers so one score does not hide either prediction quality or trading economics.

## Walk-forward selection report

[Walk-forward sweeps](Experiment-Manifest.md#walk-forward-sweeps) fit each fold independently, purge rows before validation and test, and exclude embargoed rows from later training pools. Choose the purge to cover the target's forward label horizon. Each successful trial retains its test net-return track in `trial_returns.parquet`.

At sweep conclusion, Limen reads that artifact and writes `acceptance_report.json` and `acceptance_report.md` beside it. Without the artifact, no report is written. The report computes two statistics over the recorded per-trial out-of-sample return matrix:

- **Deflated Sharpe probability:** test the selected trial's observed per-bar Sharpe against the expected maximum from the evaluated trial count and trial-Sharpe variance, with skewness and kurtosis corrections. This applies equation 2 of [Bailey and López de Prado (2014)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
- **Probability of backtest overfitting (PBO):** partition the return matrix into contiguous blocks and evaluate every balanced in-sample/out-of-sample block combination. Select the in-sample Sharpe winner, rank it out of sample, and count how often its relative rank falls in the bottom half. This follows the CSCV construction in [Bailey, Borwein, López de Prado and Zhu (2015)](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

The report's winner has the highest full-track per-bar Sharpe; exact ties prefer persisted trial order. The successful track count and the sample variance of trial Sharpes supply DSR's deflation inputs. Its PBO uses two contiguous equal blocks and evaluates both balanced combinations, so a finite report PBO can only be `0`, `0.5` or `1`. This is a minimal CSCV assessment with limited resolution; the public PBO function accepts other valid even block counts.

The Sharpe units are per bar, with no annualization. The report uses net returns already recorded by execution; it adds no position replay or new cost assumptions. Optional [acceptance thresholds](Experiment-Manifest.md#acceptance-thresholds) produce verdicts in the report. A failed threshold verdict records a research result and never aborts the run. Without thresholds, the report still contains the statistics. Degenerate or unavailable statistics are reported as `null` with an explanation; any declared verdict using them is also `null`.

The public metrics are `limen.metrics.deflated_sharpe_ratio(returns, n_trials=..., trial_sharpe_variance=...)` and `limen.metrics.probability_of_backtest_overfitting(returns_matrix, n_blocks=...)`; the matrix has one row per trial and one column per bar. Sharpe uses the arithmetic mean divided by sample standard deviation (`ddof=1`). DSR requires at least four finite returns with positive variance. Its skewness and Pearson kurtosis use centered returns standardized by population standard deviation; excess kurtosis is Pearson kurtosis minus three. The supplied trial-Sharpe variance must be finite and nonnegative. One trial or zero trial-Sharpe variance sets the comparison benchmark to zero.

PBO requires at least two trials, an even block count of at least two, exact divisibility of the track length, and at least two bars per block. Every trial must have positive finite variance in every compared half. In-sample ties select the first input trial. Out-of-sample ties use average ascending ranks divided by `n_trials + 1`; relative ranks at or below `0.5` count toward PBO. Degenerate inputs raise rather than producing a hollow probability.

These statistics reuse the sweep's test observations to assess selection. They are not a second untouched holdout. The trial count includes successfully recorded tracks in this artifact; it omits failed trials and earlier unrecorded searches and estimates no effective independent trial count. Purging and embargo depend on the declared geometry and cannot repair a feature that already reads future data. DSR's trial-count approximation and moment correction do not establish independence, stationarity or a calibrated forecast of live success. CSCV redistributes recorded blocks and is not a new chronological walk-forward evaluation. Correlated trials, serial dependence and changing market regimes limit interpretation; passing verdicts establish neither causal validity nor future profitability.

## Choosing `x`

`permutation_confusion_metrics()` and `experiment_confusion_metrics()` work on a chosen column `x`.

The default analysis column is:

```python
'price_change'
```

because it is available in the reconstructed prediction-performance table and gives a straightforward economic interpretation.

Other numeric columns are valid when they exist in the reconstructed round table and match the analysis question.

## Outlier handling

Single-round confusion summaries support outlier handling through:

- `outlier_quantiles=(lo, hi)`
- `outlier_mode='filter'` or `'winsor'`

This protects the TP/FP comparison from domination by extreme outcome values.

## Read next

- Continue to [Backtest](Backtest.md) to see how benchmark-quality signals translate into long-only trading economics.
- Continue to [Log](Log.md) for the broader post-run workflow around benchmark, backtest, and parameter correlation.
