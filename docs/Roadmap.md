# Roadmap

This page documents what Limen intends to deliver over the next twelve months (2026-07 through 2027-06), what it intends not to do, and how this intent is kept current. Limen is a Bitcoin alpha research engine; the ranking below orders work by its contribution to credible alpha discovery per unit of research time, because that is the value the downstream Vaquum stack (Nexus, Praxis, Veritas) consumes.

The issue states below were reconciled on 2026-10-20. Planned items link to issues that were open at that review; the linked issue remains authoritative afterwards. Items marked *to be filed* are committed intent without a filed issue yet; they enter delivery through the same PRD and slice process as everything else. A roadmap entry is intent, not a delivery date: work reaches `main` only through a filed slice, its gates, and review.

## Horizon key

| Tag | Meaning |
| --- | --- |
| Near | targeted for 2026-H2 |
| Sustained | continuous through the whole horizon |
| Later | targeted for 2027-H1, sequenced behind Near work |

## 1. Research integrity and statistical rigor (Near)

Research integrity remains the first priority. Chronological walk-forward evaluation with declared purge/embargo gaps, validation-return objectives and advisory DSR/PBO reports are now present. Advisory reports describe retained evidence; they do not automatically accept or reject a research result. The remaining work below must be read against those delivered contracts.

| Item | Anchor |
| --- | --- |
| Per-round exception isolation in the UEL without losing research integrity | [#602](https://github.com/Vaquum/Limen/issues/602) |
| Further statistical acceptance policy beyond advisory DSR/PBO reports | to be filed |

## 2. Leakage controls and input hygiene (Near)

| Item | Anchor |
| --- | --- |
| Deterministic regime classification columns | [#504](https://github.com/Vaquum/Limen/issues/504) |
| Explicit snapshot backtest support for multiclass and multilabel outputs | [#490](https://github.com/Vaquum/Limen/issues/490) |
| Constraint checks on all indicator and feature input parameters | [#252](https://github.com/Vaquum/Limen/issues/252) |

## 3. Throughput and scale (Near)

| Item | Anchor |
| --- | --- |
| Saving the UEL object | [#224](https://github.com/Vaquum/Limen/issues/224) |

## 4. Data layer consolidation (Sustained)

| Item | Anchor |
| --- | --- |
| Replace remaining pandas usage with polars and drop pandas from dependencies | [#643](https://github.com/Vaquum/Limen/issues/643) |
| Retype the pyarrow facade seam when the pin bumps past 23 | [#680](https://github.com/Vaquum/Limen/issues/680), [#681](https://github.com/Vaquum/Limen/issues/681) |
| Cohorts, indicator and feature taxonomy, and HistoricalData refactor | [#369](https://github.com/Vaquum/Limen/issues/369) |
| Unified market taxonomy for indicators and features | [#346](https://github.com/Vaquum/Limen/issues/346) |

## 5. Model and research surface (Later)

| Item | Anchor |
| --- | --- |
| Full TabPFN parameter surface, wider SFD parameters, and updated calibration path | [#444](https://github.com/Vaquum/Limen/issues/444) |
| Tradeline deferred remainder: multiclass and directional-conditional scope | [#605](https://github.com/Vaquum/Limen/issues/605) |

## 6. Experiment formalisms (Later)

| Item | Anchor |
| --- | --- |
| Formalizing perturbation strategies in experiments | [#341](https://github.com/Vaquum/Limen/issues/341) |
| Formalizing pruning strategies in experiments | [#344](https://github.com/Vaquum/Limen/issues/344) |
| Formalizing linkage strategies in experiments | [#342](https://github.com/Vaquum/Limen/issues/342) |
| Comprehensive reference architectures and their foundational SFDs | [#297](https://github.com/Vaquum/Limen/issues/297) |

## 7. Platform trust and supply chain (Sustained)

| Item | Anchor |
| --- | --- |
| Release provenance and supply-chain trust audit tracker | [#626](https://github.com/Vaquum/Limen/issues/626) |
| Test authority and required gates audit tracker | [#625](https://github.com/Vaquum/Limen/issues/625) |
| Manifest and YAML contract integrity audit tracker | [#632](https://github.com/Vaquum/Limen/issues/632) |
| Split the TabPFN model-download proof from the gate tracker | [#628](https://github.com/Vaquum/Limen/issues/628) |
| Master register burn-down for the 2026 audit findings | [#693](https://github.com/Vaquum/Limen/issues/693) |
| Property-based testing and fuzzing for parser and frame-contract surfaces | to be filed |

## Delivered foundations

The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) exercise current selection, seeded search, recovery, walk-forward and inference contracts. Their `execution` scenarios use recorded observations; the Compiled project also binds a validation-return objective and recorded reconstruction. These worked cases describe delivered behavior, rather than the roadmap's future intent.

The following former roadmap items have closed issues. Read the linked disposition for each item's exact delivered or retired scope:

| Former item | Recorded disposition |
|---|---|
| Deterministic standard-path enumeration when `random_search=False` | [#470](https://github.com/Vaquum/Limen/issues/470) |
| Declarable experiment objective shared across threshold optimizer, feedback, and analysis | [#533](https://github.com/Vaquum/Limen/issues/533) |
| UEL experiment conclusion report | [#211](https://github.com/Vaquum/Limen/issues/211) |
| Sensitivity analysis for parameter sweeps | [#213](https://github.com/Vaquum/Limen/issues/213) |
| `source_columns` attribute on the target interface for explicit column exclusion | [#515](https://github.com/Vaquum/Limen/issues/515) |
| Stationary shim layer for PCA-ready features | [#518](https://github.com/Vaquum/Limen/issues/518) |
| Explicit handling of probability-style snapshot inputs | [#466](https://github.com/Vaquum/Limen/issues/466) |
| Cache fitted manifest prep state deterministically when `prep_each_round=True` | [#484](https://github.com/Vaquum/Limen/issues/484) |
| Initialize and load historical data once per session | [#461](https://github.com/Vaquum/Limen/issues/461) |
| `experiment_log` as the sole canonical downstream surface, without per-permutation UEL state retention | [#485](https://github.com/Vaquum/Limen/issues/485) |
| Move polars-to-sklearn handoff off the deprecated dataframe interchange protocol | [#689](https://github.com/Vaquum/Limen/issues/689) |
| Fold XGBoost research extensions into core Limen | [#446](https://github.com/Vaquum/Limen/issues/446) |
| Formalizing search strategies in experiments | [#343](https://github.com/Vaquum/Limen/issues/343) |

[Walk-forward evaluation](Walk-Forward.md), [calibration and objectives](Calibration.md) and [selection reports](Benchmark.md#walk-forward-selection-report) describe the corresponding current boundaries.

## What Limen will not do

These are deliberate boundaries, not backlog. They mirror the product boundary in the [README](../README.md) and the disclaimed scope in [Benchmark](Benchmark.md):

- No venue order submission or downstream trading decisions; Limen's research execution engine evaluates declared policies, while Nexus and Praxis own operational decisions and orders.
- No generic multi-asset research platform; Limen stays Bitcoin-native.
- No upstream source-of-truth market data infrastructure; Origo owns the data layer.
- No synthetic market data generation.
- No public leaderboard, benchmark corpus, or model card program.
- No hosted service or graphical interface; Limen remains a Python library and CLI.

## How this roadmap stays current

The roadmap is reviewed whenever a PRD is filed and at least quarterly; changes land by pull request under the same documentation gates as every other page. Priority order may change between reviews; the anchored issues carry the authoritative current state of each item.

## Read next

- [Docs hub](README.md)
- [Benchmark](Benchmark.md)
- [Universal Experiment Loop](Universal-Experiment-Loop.md)
- [Developer home](Developer/README.md)
