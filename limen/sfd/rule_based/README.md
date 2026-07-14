# `limen.sfd.rule_based`

> Configuration schema and predicate compiler for non-learned, rule-based decoders.

## Canonical docs

- [Single-File Decoder](../../../docs/Single-File-Decoder.md)

## What this package owns

Owns the rule-based decoder building blocks: the `RuleBasedConfig` schema (with leaf-condition validation) and `build_predicate`, which compiles a condition tree into an evaluable predicate over a frame.
Does **not** own the SFD wiring that uses these (owned by the `rule_based` modules in `limen.sfd.foundational_sfd` / `reference_architecture`) or experiment execution.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `RuleBasedConfig` | Validated rule-strategy configuration | Enforces required fields per leaf condition type |
| `build_predicate` | Compile a condition tree into a callable predicate | Evaluated against a `pl.DataFrame` |

## Adjacent modules

- `limen.sfd.reference_architecture.rule_based` (`RuleBasedStrategy`) and `limen.sfd.foundational_sfd.rule_based` build on these.
- `limen.data.utils.split_data_to_rule_based_prep_output` produces the `data_dict` variant that keeps the raw columns predicates reference.

## Quick orientation

```text
rule_based/
├── config.py       # RuleBasedConfig + leaf-condition validation
└── predicates.py   # build_predicate + operator table
```

## Things to know

- Leaf conditions are validated by type: each of `threshold`, `relative`, `crossover`, `slope`, and `sql_expr` has its own required fields, and a missing field raises `ValueError`.
- Logical nodes combine leaves with `and`, `or`, and `not`.
- A condition may specify `persistence_n` **or** `recency_n`, but not both.
- Predicate values support `{param}`-style formatting, so thresholds can be parameterized from the manifest.

## Read next

- [Single-File Decoder](../../../docs/Single-File-Decoder.md)
- [Reference Architecture](../../../docs/Reference-Architecture.md)
