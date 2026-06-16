# Benchmark

In Limen, benchmark is the prediction-quality layer between the raw experiment log and the trading backtest.

It measures signal activity, positive-class accuracy, and true-positive versus false-positive outcome separation. Benchmark comes before PnL compression so prediction quality remains visible before a trading rule is applied.

## Risk boundary

Benchmark output is research evidence, not investment advice, trading advice, or a promise of future performance. A benchmark table can show statistical structure without proving that a strategy survives live execution, fees, slippage, or portfolio constraints.

## Where benchmark lives

Benchmark analytics are built on top of `Log`.

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
