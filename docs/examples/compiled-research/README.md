# Compiled Research project

The Manifest owns preparation and its configured checks. Trainer reconstructs selected recorded rounds and validates their metrics.

The [research chapter](../../Research-Book.md) develops the project. The [documentation contract](../../Developer/Documentation-System.md) defines the proof required before this candidate is published as the complete book.

## Prerequisites

Use the project and input bundle identified by the example evidence, Python 3.10–3.13, and its matching Limen wheel. Selection needs the data extra. The full project needs `vaquum-limen[data,stats,boosting]`; Sobol uses statistics; factorization uses LightGBM.

## Run the selection scenario

Install the matching `vaquum-limen[data]` distribution. From the example bundle root, supply the directory containing the recorded input files:

```bash
python compiled-research/research.py --scenario selection --data-dir inputs --output-dir results/compiled
```

From a repository checkout, work in `docs/examples/`, use `--data-dir ../../tests/fixtures` and put generated output under `../../.docs-evidence/`. The command checks the recorded file digest before research. It writes `summary.json`, `results.csv`, `metadata.json` and the compressed round artifacts. Trainer reconstructs the selected round from those artifacts and validates the replayed metrics; it does not restore saved model weights.

## Continue through the research decisions

`--scenario all` runs the named scenarios in separate output directories. `--scenario` also accepts one name from the table. Use a new output directory for a fresh run; the recovery scenario creates its interrupted run and resumes it internally.

| Scenario | Decision and output |
|---|---|
| `search` | Compare grid, seeded random, Sobol and shuffled traversal over a bounded domain. |
| `recovery` | Interrupt after two rounds, retain a checkpoint, resume without repeating a completed round. |
| `walk_forward` | Keep each test window after its fitting/validation observations, with purge and embargo. |
| `holdout` | Assess final observations only after selection; retain the assessment and its exposure limits. |
| `calibration` | Compare all four calibration/threshold modes and disclose validation reuse. |
| `execution` | Execute recorded observations under explicit costs, product rules and causal clocks. |
| `bars` | Form volume, trade and liquidity bars; omit an incomplete final bar. |

Factorized search uses the admitted deterministic LightGBM grid with independent fee and tree-count axes. The ordinary search modes use logistic regression; these are different declared comparisons. The walk-forward scenario reconstructs qualified folds and binds their complete identities into a Cohort. `allocation` adds batch meta-allocation with completed-opportunity training; it makes no live-promotion or recovery claim.

The holdout scenario creates a separate Git project, reserves its final window before research, freezes selected fitted state and retains the final report. Local recorded files leave earlier exposure unknown. Preserve its project marker, history and assessment artifacts together.

Real TabPFN model validation remains unproved under [#628](https://github.com/Vaquum/Limen/issues/628).
