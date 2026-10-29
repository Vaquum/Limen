# Interpret a prediction as a trade

A forecast does not determine a return until entry, exit, exposure, prices and costs are declared. Start from the policy used to evaluate the candidate; changing that policy changes the question.

The execution variants of the [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) use recorded hourly observations and a separately pinned funding history. Their ledgers state the actual policy and evaluated window.

## Prerequisites

Start with the selected predictions and their declared availability. Use the unchanged recorded bars and funding input for the execution variant; changing the product, capital or trading policy creates a different comparison.

## The execution engine determines the accounting model

| Execution path | Position and price interpretation | Retained evidence |
|---|---|---|
| Snapshot | Lagged binary positions over aligned return and OHLC evidence, with configured costs and optional barriers | Twenty summary metrics and optional aligned execution tracks |
| Event cash spot | Actual admitted intents and fills against recorded source clocks, with cash and inventory constraints | States, intents, fills, episodes, and funding ledgers |
| Event linear perpetual | Signed or fractional exposure under declared quote-settled economics and collateral | The same ledger families plus configured funding recognition and settlement |

Snapshot multiclass outputs require an explicit binary mapping, and multilabel outputs require an explicit selected label. The limited short/flat snapshot path has its own contract. Event signed exposure instead requests a target position under the configured sizing rules. [Backtest](Backtest.md) owns the exact inputs, metric definitions, and supported policy combinations.

## Decision and fill clocks preserve different observations

A decision timestamp identifies the model's row. Availability identifies when the information needed by that decision exists. Recorded source intervals also distinguish open-price availability from full-bar availability. Execution lag applies after the eligible decision becomes available.

The event engine retains pending intents until an admissible recorded fill exists. A fine execution source or formed-bar constituent history supplies its own price and availability evidence. Those clocks explain the relationship between a requested position, its eventual fill, and its measured return.

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

## Signed exposure retains sizing and exit policy

Changed sizing signals target post-cost current equity subject to the configured quantity step, minimum notional, and collateral constraints. Unchanged sizing signals hold quantity. Resizes and reversals therefore differ from repeatedly restoring an unchanged percentage exposure on every observation.

Independent exits, elapsed holding limits, timers, take-profit, and stop-loss can close exposure before the model changes direction. Rearming governs later entry. The terminal policy marks remaining event exposure without forcing an invented close. The retained contract identifies those choices beside the ledger.

## Funding uses its own observed schedule

Funding declares rate units, valuation conventions, schedule, phase, and recognition or cash-settlement behavior. Discrete settlement and continuous accrual have separate accounting rules. The supplied Binance history provides realized rates and mark prices for the worked perpetual case.

Binance BTCUSDT, Hyperliquid BTC, and Deribit BTC/USDC presets provide declared mechanics and scenario assumptions. Supplied historical funding replaces those scenario rates and must cover the required interval. A settled funding value is cost evidence at its recorded availability; a predictive funding feature requires its own causal source.

## Observed fills can calibrate a retrospective cost comparison

The Python-only observed-cost scenario freezes a calibration artifact from pinned actual fills and reference observations. It evaluates each candidate with its original and calibrated execution policy. The admitted calibration changes slippage. Candidate fees remain the declared fee schedule.

<!-- example: trading-python-observed-costs source=docs/examples/python-research/scenarios.py region=observed_costs scenario=observed_costs interface=python role=executed -->
```python
def observed_costs(data_dir, output):
    fixture = data_dir / 'observed_costs'
    expected = {'history.zip': 'df4d7776172df459ec297cbcfe8e72a4748a51ab626e3c990deec83415102093',
                'benchmark.json': 'b71011697713417b21302e670abf78c0bff739ec9135c7012db02fa3b2d6ccf8'}
    for name, digest in expected.items():
        if hashlib.sha256((fixture / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Recorded execution-cost input differs: ' + name)
    with ZipFile(fixture / 'history.zip') as archive:
        history = output / 'history.json'
        history.write_bytes(archive.read('admitted.json'))
    calibration = CostCalibration.from_history(history, cutoff_ns=1785810791000000000)
    artifact, specification = output / 'calibration.json', output / 'research-specification.json'
    calibration.write(artifact)
    specification.write_text(json.dumps({'calibration_digest': calibration.digest}))
    reports = []
    for mode in ('binary', 'target_exposure'):
        for hour in (1, 2):
            experiment, comparison = run_cost_experiment(artifact, specification, fixture / 'benchmark.json',
                                                       output / f'{mode}-{hour}', mode=mode, hour=hour)
            reports.append({'mode': mode, 'hour': hour, 'rounds': project.completed_log(experiment).height,
                            'comparison': comparison.report()})
    return {'calibration': calibration.report(), 'runs': reports,
            'historical_delivery': 'unknown; retrospective stress only'}
```

<!-- result: trading-python-cost-support interface=python scenario=observed_costs fields=calibration.support,historical_delivery -->

Support, side concentration, source clocks, and historical delivery limits remain part of the comparison. The retained sample supports a retrospective stress comparison under its stated assumptions. [Observed-cost contracts](Backtest.md#recorded-execution-price-costs) define admission, immutable calibration, comparison artifacts, and unchanged candidate identity.

## Read the fills

These excerpts show the first eight native fills, with Unix nanosecond timestamps, signed quantity changes, reference prices and fees. The full ledgers remain in the native records. The two campaigns use different declared assessment populations; compare their policy and clocks before comparing a return.

### Python Research ledger

<!-- result: trading-python-ledger interface=python scenario=execution fields=ledger_excerpt[].time_ns,ledger_excerpt[].quantity_delta,ledger_excerpt[].reference_price,ledger_excerpt[].fee -->

### Compiled Research ledger

<!-- result: trading-compiled-ledger interface=compiled scenario=execution fields=execution.ledger_excerpt[].time_ns,execution.ledger_excerpt[].quantity_delta,execution.ledger_excerpt[].reference_price,execution.ledger_excerpt[].fee -->

## Carry the policy with the result

A Sensor emits a prediction with its availability and identity. Downstream trading owns order execution. Continue to [linked learning stages](Linked-Models.md) with the declared timing and economics, then [research evidence](Research-Evidence.md) to interpret the resulting candidate scores.
