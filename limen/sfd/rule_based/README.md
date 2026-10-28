# `limen.sfd.rule_based`

> Configuration schema and predicate compiler for non-learned, rule-based decoders.

## Canonical docs

- [Rule-Based Strategy](../../../docs/Experiment-Manifest.md#rule-based-strategy) — conditions, temporal modifiers and parameter references
- [Single-File Decoder](../../../docs/Single-File-Decoder.md)

## In the research projects

The [Python project](../../../docs/examples/python-research/README.md) supplies its own execution signals. The [Compiled project](../../../docs/examples/compiled-research/README.md) `allocation` scenario includes a declared rule-based primary. In both paths, the execution contract determines fills, costs and availability after a research decision has been formed.

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

| Module or family | Responsibility |
|---|---|
| `config.py` | RuleBasedConfig + leaf-condition validation |
| `predicates.py` | build_predicate + operator table |

<!-- reference: package-sfd-rule_based-layout source=limen/sfd/rule_based/README.md region=layout:package role=layout -->
```text
rule_based/
├── README.md
├── __init__.py
├── _conditions.py
├── _temporal.py
├── common_strategies.py
├── config.py
└── predicates.py
```

## Things to know

Selected Compiled rule pipelines can be reconstructed through Trainer and consumed by Sensor and binary Cohort. See the canonical [selected-rule inference contract](../../../docs/Trainer.md#selected-rule-rounds), including identical-window comparisons and caller-owned SQL causality.

- Leaf conditions are validated by type: each of `threshold`, `relative`, `crossover`, `slope`, and `sql_expr` has its own required fields, and a missing field raises `ValueError`.
- Logical nodes combine leaves with `and`, `or`, and `not`.
- Leaf conditions may specify `persistence_n` **or** `recency_n`; see the [canonical temporal-modifier contract](../../../docs/Experiment-Manifest.md#temporal-modifiers).
- Predicate values support `{param}`-style formatting, so thresholds can be parameterized from the manifest.

## Read next

- [Single-File Decoder](../../../docs/Single-File-Decoder.md)
- [Reference Architecture](../../../docs/Reference-Architecture.md)
