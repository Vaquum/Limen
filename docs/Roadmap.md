# Research priorities and delivered capabilities

Limen's priorities concern credible Bitcoin research, explicit selection decisions, and reproducible inference. The roadmap records remaining work alongside delivered source capabilities. Package releases and their execution evidence establish which source revision is available in a published wheel.

Issue states were reviewed on 29 October 2026. The linked issue records the exact delivered, deferred, or retired scope. Closed research trackers can contain bounded implementations and explicit exclusions; their disposition remains part of the contract.

## Delivered research capabilities have current contracts

The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) exercise separately authored research definitions over recorded observations. The book develops their decisions, and Reference supplies exact API and configuration contracts.

| Capability | Delivered source scope | Contract and disposition |
|---|---|---|
| Research interfaces | Caller-owned Python methodology and YAML/CLI Compiled Research | [Research book](Research-Book.md), [interface contract](../README.md#two-interfaces-different-purposes) |
| Features and market state | Native indicators, feature families, validated parameters, and a stateful seven-label regime classifier | [Feature design](Feature-Design.md), [#504](https://github.com/Vaquum/Limen/issues/504), [#252](https://github.com/Vaquum/Limen/issues/252), [#346](https://github.com/Vaquum/Limen/issues/346) |
| Model families | Binary, sequence, regression, quantile, linked-learning, and rule architectures with model-specific restrictions | [Model choice](Model-Selection.md), [#297](https://github.com/Vaquum/Limen/issues/297), [#369](https://github.com/Vaquum/Limen/issues/369) |
| TabPFN controls | Supported classifier parameters and separately retained real-checkpoint execution | [Optional model campaign](examples/compiled-research/README.md#optional-model-campaign), [#444](https://github.com/Vaquum/Limen/issues/444), [#628](https://github.com/Vaquum/Limen/issues/628) |
| Perturbation | Population, feature, representation, target, and model axes | [Robustness research](Robustness-Research.md), [#341](https://github.com/Vaquum/Limen/issues/341) |
| Search and feedback | Grid, random, Sobol, mutable queues, five reducers, and correlation/forest analysis | [Search](Advanced-Search.md), [feedback](Reducers-And-Feedback.md), [#343](https://github.com/Vaquum/Limen/issues/343), [#344](https://github.com/Vaquum/Limen/issues/344) |
| Linked learning | Bounded residual, feature, weight, pseudo-label, and direction/sizing mechanisms | [Linked models](Linked-Models.md), [#342](https://github.com/Vaquum/Limen/issues/342) |
| Captured state and failure policy | Trusted `.uel` snapshots and restricted budgeted serial candidate-error recording | [Run and recover](Run-And-Recover.md), [#224](https://github.com/Vaquum/Limen/issues/224), [#602](https://github.com/Vaquum/Limen/issues/602) |
| Analysis tables | Polars result tables with explicit correlation grouping columns | [Log](Log.md), [#643](https://github.com/Vaquum/Limen/issues/643) |
| Execution interpretation | Snapshot/event policies, explicit multiclass/multilabel mapping, independent exits, funding, and observed-cost comparisons | [Trading behavior](Trading-Behavior.md), [#490](https://github.com/Vaquum/Limen/issues/490) |
| Chronological assessment | Walk-forward fits, retained return tracks, advisory DSR/PBO, and experiment-family reports | [Walk-forward](Walk-Forward.md), [research evidence](Research-Evidence.md) |
| Selected behavior | Python fitted-state adapters, native reconstruction, compatible Cohorts, and Bayesian Python voting | [Selected behavior](Selected-Behavior.md) |
| Protected research | Reservation-bound exclusion, supported workflow composition, frozen selection, and restricted final assessment | [Final holdout](Final-Holdout.md), [workflow compatibility](Experiment-Manifest.md#supported-research-workflows) |
| Configuration integrity | Exact schema version, structural and callable validation, parameter consumption, and source-bound artifacts | [Manifest](Experiment-Manifest.md), [#632](https://github.com/Vaquum/Limen/issues/632) |
| Tradeline scope | Recorded disposition of multiclass and directional-conditional research requests | [#605](https://github.com/Vaquum/Limen/issues/605) |

## Remaining work follows its recorded authority

| Priority | Remaining work | Authority |
|---|---|---|
| Research integrity | Statistical acceptance policy beyond the current advisory reports | A new issue or PRD must define the additional scientific claim and its evidence. |
| Platform trust | Release provenance and supply-chain audit follow-up | [#626](https://github.com/Vaquum/Limen/issues/626) |
| Test authority | Required-gate and test-authority audit follow-up | [#625](https://github.com/Vaquum/Limen/issues/625) |
| Audit disposition | Remaining findings in the master register | [#693](https://github.com/Vaquum/Limen/issues/693) |
| Dependency maintenance | Retype the Arrow facade seam when the pinned dependency passes version 23 | [#681](https://github.com/Vaquum/Limen/issues/681), following [#680](https://github.com/Vaquum/Limen/issues/680) |
| Verification methods | Further parser and frame-contract property testing or fuzzing | Additional scope requires a filed issue or PRD; existing checks remain in force. |

An open tracker identifies work requiring disposition. Its presence does not make every listed historical finding an unimplemented product capability. Delivery follows the repository's issue, review, test, and release requirements.

## Product boundaries assign downstream responsibility

Limen researches Bitcoin models and strategies through Python functions and YAML/CLI definitions. Origo owns upstream source-of-truth market-data infrastructure. Nexus and Praxis own downstream operational decisions and venue orders. The [execution reference](Backtest.md) specifies the research accounting supported inside Limen.

The documentation reader and prose editor support the book's reading and contribution workflows. Their deployments have separate evidence and activation requirements, described in [Edit and verify the book](Developer/Editing-The-Book.md).

Historical Explorer, Rust, transformer, and experimental parallel branches retain their own revision and development status. Current behavior belongs in the book only when it exists in the released source and has its stated execution evidence.

## Roadmap changes preserve delivered meaning

Roadmap review accompanies a new PRD or a material delivery and occurs at least quarterly. A completed item moves to its delivered contract with the exact disposition retained. New intent identifies its issue, prerequisite, and evidence requirement before it becomes a delivery commitment.

The [docs hub](README.md) provides the reading sequence. [Developer home](Developer/README.md) provides contribution and release procedures.
