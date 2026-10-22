# Interpret a prediction as a trade

A forecast does not determine a return until entry, exit, exposure, prices and costs are declared. Start from the policy used to evaluate the candidate; changing that policy changes the question.

The execution variants of the [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) use recorded hourly observations and a separately pinned funding history. Their ledgers state the actual policy and evaluated window.

## Python Research

The base experiment trades the next bar from open to close. A positive forecast takes positive exposure, a negative forecast takes negative exposure, and each nonzero decision pays two basis points round trip. The returned value is the mean net return per evaluated observation:

<!-- example: trading-python-score source=docs/examples/python-research/research.py region=validation_score scenario=selection interface=python role=executed -->
```python
def backtest(validation, forecasts):
    positions = np.sign(forecasts)
    net = positions * validation["next_return"].to_numpy() - abs(positions) * 0.0002
    return float(np.mean(net))
```

That caller-owned policy is deliberately small. The `execution` variant separately uses explicit event inputs, product and funding rules. It retains actual fills and account states rather than treating the forecast itself as a fill.

## Compiled Research

The worked research manifest declares the product, source interval and execution assumptions alongside the model:

<!-- example: trading-compiled-policy source=docs/examples/compiled-research/research.yaml region=execution scenario=search interface=compiled role=executed -->
```yaml
    backtest:
      source_interval_seconds: 900
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
      execution_lag_seconds: 0.0
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1.0e-9
        min_notional: 0.0
```

Read the event ledger in the retained native records: each fill has a time, quantity, price and fee; account states retain exposure and equity. The `execution` and `funded` variants change the declared economics and retain their own results:

<!-- example: trading-compiled-variants source=docs/examples/compiled-research/run.sh region=execution scenario=execution interface=compiled role=executed -->
```bash
limen run --no-progress-bar \
  compiled-research/execution.yaml \
  compiled-research/funded.yaml
```

A snapshot, a cash account and a perpetual account are different interpretations. Compare returns only after the observation population, availability, policy and capital conventions agree. The [execution reference](Backtest.md) preserves every supported product, clock, barrier, funding, allocation and observed-cost contract. The [benchmark reference](Benchmark.md) defines when retained tracks are comparable.

## Read the fills

These excerpts show the first eight native fills, with Unix nanosecond timestamps, signed quantity changes, reference prices and fees. The full ledgers remain in the native records. The two campaigns use different declared assessment populations; compare their policy and clocks before comparing a return.

### Python Research ledger

<!-- result: trading-python-ledger interface=python scenario=execution fields=ledger_excerpt[].time_ns,ledger_excerpt[].quantity_delta,ledger_excerpt[].reference_price,ledger_excerpt[].fee -->

### Compiled Research ledger

<!-- result: trading-compiled-ledger interface=compiled scenario=execution fields=execution.ledger_excerpt[].time_ns,execution.ledger_excerpt[].quantity_delta,execution.ledger_excerpt[].reference_price,execution.ledger_excerpt[].fee -->

## Carry the policy with the result

A Sensor emits a prediction with its availability and identity; it does not submit venue orders or execute research exits. Downstream trading still owns execution. Continue to [selected behavior](Selected-Behavior.md) with the fitted state and the exact policy under which it was selected.
