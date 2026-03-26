# Benchmark

In Limen, benchmark refers to the prediction-quality layer that sits between raw experiment results and trading backtests. It helps answer whether a decoder is making useful directional calls before you translate those calls into simulated returns.

## Current Surface

The benchmark surface is exposed through the `Log` analytics:

- `uel.experiment_confusion_metrics`
- `uel._log.experiment_confusion_metrics('price_change')`
- `uel._log.permutation_confusion_metrics(x='price_change', round_id=...)`

## What It Measures

These tables focus on classification quality and outcome separation, including:

- predicted-positive rate
- actual-positive rate
- precision
- recall
- TP and FP counts
- mean and median of a chosen outcome column `x`
- separation between TP and FP distributions through Cohen's d and KS

In practice, this lets you inspect not just whether a model calls LONG correctly, but whether true positives also look materially better than false positives on the metric you care about.

## Why It Exists

Benchmark analytics are intentionally distinct from backtests:

- benchmark asks whether the signal has predictive structure
- backtest asks whether that structure survives simple trading assumptions and costs

That separation makes it easier to understand why a permutation looks good or bad, instead of collapsing everything into a single return number too early.
