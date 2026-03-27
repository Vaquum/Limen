# Benchmark

In Limen, benchmark is the prediction-quality layer between the raw experiment log and the trading backtest.

It answers questions like:

- is the signal calling positives at a sensible rate?
- when it calls positive, is it usually right?
- do true positives look materially better than false positives on the outcome we care about?

That is a different question from "does this make money under a trading rule?" Benchmark comes first so you can understand the signal before you compress everything into PnL.

## Where Benchmark Lives

Benchmark analytics are built on top of `Log`.

The main surfaces are:

- `uel.experiment_confusion_metrics`
- `uel._log.experiment_confusion_metrics('price_change')`
- `uel._log.permutation_confusion_metrics(x='price_change', round_id=0)`

The first two are the same analysis surfaced in two places: UEL computes the experiment-wide confusion table automatically at the end of a run.

## Typical Benchmark Workflow

```python
benchmark = uel.experiment_confusion_metrics

round0 = uel._log.permutation_confusion_metrics(
    x='price_change',
    round_id=0,
)
```

Use the experiment-wide table to compare rounds. Use the single-round table when one permutation deserves a closer look.

## What Benchmark Measures

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

When `x='price_change'`, the table is asking:

- how often did the model call long?
- how often was long actually the right class?
- among the bars the model called long, how different were the true positives from the false positives on realized price change?

## How To Read It

### Signal rate

`pred_pos_rate_pct` tells you how active the signal is. A round with strong precision but a trivially small positive rate may not be operationally interesting.

### Precision and recall

- `precision_pct` asks: when the model predicted positive, how often was it right?
- `recall_pct` asks: when the true class was positive, how often did the model catch it?

You usually want to interpret these together. High precision with very low recall means a selective signal. High recall with weak precision means a noisy signal.

### TP versus FP quality

The most useful part of Limen's benchmark layer is that it does not stop at precision and recall.

- `tp_x_mean` and `tp_x_median` describe the chosen outcome inside true positives
- `fp_x_mean` and `fp_x_median` do the same for false positives
- `tp_fp_cohen_d` and `tp_fp_ks` estimate how separated those two distributions are

If TP and FP are not meaningfully separated on the outcome you care about, a round may be statistically correct without being especially useful.

## Benchmark Versus Backtest

Benchmark and backtest are intentionally separate:

- benchmark asks whether the signal has predictive structure
- backtest asks whether that structure survives a concrete long-only trading rule with costs

This separation matters because the layers often disagree.

Common cases:

- a round can have attractive benchmark metrics but weak trading economics
- a round can have modest benchmark metrics yet produce decent backtest behavior because of position profile and cost structure

That is why Limen exposes both layers instead of forcing you to choose a single score too early.

## Choosing `x`

`permutation_confusion_metrics()` and `experiment_confusion_metrics()` work on a chosen column `x`.

The most common choice is:

```python
'price_change'
```

because it is available in the reconstructed prediction-performance table and gives a straightforward economic interpretation.

You can use other numeric columns when they exist in the reconstructed round table and match the question you are asking.

## Outlier Handling

Single-round confusion summaries support outlier handling through:

- `outlier_quantiles=(lo, hi)`
- `outlier_mode='filter'` or `'winsor'`

This is useful when a few extreme outcome values would otherwise dominate the TP/FP comparison.

## Read Next

- Continue to [Backtest](Backtest.md) to see how benchmark-quality signals translate into simple trading economics.
- Continue to [Log](Log.md) for the broader post-run workflow around benchmark, backtest, and parameter correlation.
