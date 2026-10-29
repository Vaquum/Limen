# Rules make the strategy hypothesis explicit

A rule strategy maps observed conditions to requested positions. Thresholds, crossovers, persistence, and exits express that hypothesis directly. Parameter search then compares declared alternatives under the same observation and execution policy.

The [Python project](examples/python-research/README.md) evaluates ordinary Polars expressions. The [Compiled project](examples/compiled-research/README.md) declares a validated condition graph in YAML. The worked cases use the same hourly assessment interval and two price floors, allowing their Boolean decisions to be checked independently.

## Prerequisites

Both projects require the matching wheel and recorded-input bundle. The rule cases use the `data` extra. The observation clock, feature history, and intended execution policy must be established before interpreting a position as a trade.

## Predicate composition defines the entry

The worked hypothesis requires a close above a declared price floor, recent or persistent upward activity, and a rise over three observations. The current bar must also close at or above its open. The comparison varies the floor between 40,000 USDT and 41,000 USDT per BTC.

| Condition | Meaning in this case |
|---|---|
| Threshold | Current close reaches the declared price floor. |
| Relative comparison | Close exceeds open on the same bar. |
| Persistence | The relative condition holds for two consecutive bars. |
| Recency | A close/open upward cross occurred within three bars. |
| Slope | Close exceeds the close three bars earlier. |
| Composition | Persistent or recent activity combines with the floor, slope, and current-bar condition. |

These predicates evaluate completed hourly observations. Their signals become available after the current close. The execution policy determines the admissible later fill.

## Python expresses the predicates as functions

The Python case computes the Boolean conditions on the declared assessment partition. Rolling history starts at that partition's beginning, and early unavailable comparisons become false:

<!-- example: rules-python-predicates source=docs/examples/python-research/rule_cases.py region=rule_signals scenario=search interface=python role=executed -->
```python
def evaluate_rules(raw):
    bars = raw.filter(pl.col('datetime').is_between(
        pl.datetime(2024, 1, 18, time_zone='UTC'), pl.datetime(2024, 1, 22, time_zone='UTC'), closed='left'))
    up = pl.col('close') > pl.col('open')
    cross = up & (pl.col('close').shift(1) <= pl.col('open').shift(1))
    persistent = (up.cast(pl.Int8).rolling_sum(2, min_samples=2) == 2).fill_null(False)
    recent = (cross.fill_null(False).cast(pl.Int8).rolling_sum(3, min_samples=1) >= 1)
    rising = (pl.col('close') > pl.col('close').shift(3)).fill_null(False)
    rows = []
    for floor in (40000., 41000.):
        signal = bars.select(((pl.col('close') >= floor) & (persistent | recent) & rising &
                              ~(pl.col('close') < pl.col('open'))).cast(pl.Int8).alias('signal'))['signal']
        rows.append({'price_floor': floor, 'signals': signal.to_list(), 'positive_decisions': int(signal.sum())})
    return {'rows': rows, 'assessment_rows': bars.height, 'history_resets_at_split': True}
```

The returned signal tape and positive-decision count describe the rule's outputs. A trading comparison additionally needs entry, exit, price, and cost conventions. The project's separate baseline campaign applies its explicit next-open-to-close policy.

## YAML names the graph and its parameter domain

The Compiled definition assigns each condition an identity and combines them through named operands. Validation checks references, duplicate identities, dependencies, and cycles before execution:

<!-- example: rules-compiled-predicates source=docs/examples/compiled-research/rule_contracts.yaml region=declaration scenario=search interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: book_rule_contracts
  mode: development
sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_dates:
      train_start: '2024-01-01'
      train_end: '2024-01-15'
      val_start: '2024-01-15'
      val_end: '2024-01-18'
      test_start: '2024-01-18'
      test_end: '2024-01-22'
    strategy:
      conditions:
      - {id: floor, name: Close above the declared floor, type: threshold, column: close, operator: '>=', value: '{price_floor}'}
      - {id: persistent_up, name: Two up bars, type: relative, column: close, operator: '>', other_column: open, persistence_n: 2}
      - {id: recent_cross, name: Recent close-open cross, type: crossover, column: close, other_column: open, direction: above, recency_n: 3}
      - {id: rising, name: Close above three bars ago, type: slope, column: close, direction: rising, lookback: 3}
      - {id: down, name: Current down bar, type: sql_expr, expr: 'close < open'}
      - {id: not_down, name: Exclude down bars, operator: not, operands: [down]}
      - {id: activity, name: Persistent or recent activity, operator: or, operands: [persistent_up, recent_cross]}
      - {id: entry, name: Combine the entry conditions, operator: and, operands: [floor, activity, rising, not_down]}
      entry: entry
    reference_architecture: limen.sfd.reference_architecture.rule_based
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
  params:
    price_floor: [40000.0, 41000.0]
uel:
  output_path: compiled-rule-contracts
  n_permutations: 2
  search_strategy: {type: grid}
  output_format: csv
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
```

The `sql_expr` leaf supplies the current down-bar comparison. SQL expressions have their own information requirements. An expression using a future row or a whole-window aggregate changes its availability even when its syntax is valid. [Manifest rule contracts](Experiment-Manifest.md#rule-based-strategy) define the supported operations and their temporal modifiers.

## The signal comparison has a declared population

Both cases assess the half-open UTC interval from 18 January 2024 to 22 January 2024. The independent assertions reconstruct the predicates from recorded prices and compare the complete signal tapes. Their counts therefore measure the same rule conditions and split-local history:

<!-- result: rule-research-python-decisions interface=python scenario=search fields=rules.assessment_rows,rules.rows[].price_floor,rules.rows[].positive_decisions -->

<!-- result: rule-research-compiled-decisions interface=compiled scenario=search fields=rules.assessment_rows,rules.rows[].price_floor,rules.rows[].positive_decisions -->

The Compiled run also records its configured backtest. The Python predicate diagnostic reports decisions without attaching a return. The [trading chapter](Trading-Behavior.md) develops how a signal becomes a costed position.

## Entry and exit can express separate hypotheses

Independent exit rules have their own condition namespace and entry reference. They support the admitted threshold, relative, crossover, slope, and compound predicates. Their contract applies to binary entries from either a rule strategy or an ML model. Exit SQL is excluded by validation.

A separate exit can close a position while its entry signal remains true. Rearming then determines when a new entry becomes eligible. The retained definition must include both policies when research is reconstructed or compared. The [independent-exit contract](Experiment-Manifest.md#independent-exit-rules) owns the exact supported paths and replay requirements.

The eight packaged common strategies supply explicit baselines: buy-and-hold, SMA, EMA, MACD, RSI, Bollinger, Donchian, and time-series momentum. Their [reference](Common-Strategies.md) specifies parameters, stateful entry/exit rules, costs, and recorded results. The dollar-bar crash-reversal strategy has separate source and availability requirements.

## Selected rules preserve their history requirements

A reconstructed rule Sensor retains the chosen predicate configuration and preparation. Rolling conditions depend on the supplied history window. Warm-up, missing features, and selection guards remain visible in inference results. The [Trainer reference](Trainer.md#selected-rule-rounds) describes same-window reconstruction and the boundary between research labels and inference inputs.

Continue to [define the experiment](Compiled-Research.md) with the entry hypothesis, parameter domain, and evaluation policy fixed. The subsequent search compares alternatives under those declarations.
