# Python Research project

Your functions own preparation, fitting, backtesting and selection. The selected Sensors reuse the evaluated fitted objects.

The [research chapter](../../Research-Book.md) develops the project. The [documentation contract](../../Developer/Documentation-System.md) defines the proof required before this candidate is published as the complete book.

## Prerequisites

Use the project and input bundle identified by the example evidence, Python 3.10–3.13, and its matching Limen wheel. Selection needs the data extra. The full project needs `vaquum-limen[data,stats]`; Sobol uses statistics.

## Run the selection scenario

Install the matching `vaquum-limen[data]` distribution. From the example bundle root, supply the directory containing the recorded input files:

<!-- example: python-selection-command source=docs/examples/commands.json region=python-selection scenario=selection interface=python role=executed -->
```bash
python python-research/research.py --scenario selection --data-dir inputs --output-dir results/python
```

From a repository checkout, work in `docs/examples/`, use `--data-dir ../../tests/fixtures` and put generated output under `../../.docs-evidence/`. The command checks the recorded file digest before research. It writes `research.csv` and `summary.json`. The evaluated models and Sensors remain in caller-owned memory during the run; those two files do not restore fitted state. Import `run_research` to continue using the returned Cohort and retained models in the same Python process.

## Continue through the research decisions

`--scenario all` runs the named scenarios in separate output directories. `--scenario` also accepts one name from the table. Use a new output directory for a fresh run; the recovery scenario creates its interrupted run and resumes it internally.

The execution scenario compares cash spot with a linear perpetual using the recorded funding history, realized rates and mark prices. The shared `recorded_inputs.py` checks and normalizes those records; the native funding ledger exposes each payment. The bar scenario also forms three CFR diagnostic bars from the pinned 90-second trade slice. This is shared preprocessing on recorded events, not a claim that the short slice can support model selection.

Search also runs a separate resource-controlled grid. `BudgetReducer(max_permutations=3)` reduces a four-candidate budget to three completed rounds through the shared feedback controller. Its audit and checkpoint retain the intervention and remaining queue state. This changes the amount of research performed; it is not evidence that the reduced search selects a better model.

Search and recovery use the first 1,440 recorded hourly observations. Recovery compares the completed resumed run with a separate uninterrupted run. Walk-forward uses the first 384 hourly observations; its inference callback declares their one-hour availability interval. Selection, ordinary final assessment and classifier calibration retain their quarter-hour population.

The observed-cost scenario creates one immutable calibration from the recorded Chainticks population and pins its digest separately. Four caller-owned fits compare original and calibrated ledgers across two hours and both binary and signed exposure. Prices are recorded midpoints encoded as points; capital and strategy exposure are software scenarios. The calibration archive postdates evaluation, historical delivery is unknown and sell support is concentrated. These are retrospective stress results. The bundle retains the source attribution, CC BY 4.0 license and reproduction files. The [Backtest reference](../../Backtest.md#frozen-candidates-in-custom-python-research) uses the exact same model source.

| Scenario | Decision and output |
|---|---|
| `search` | Compare grid, seeded random, Sobol and shuffled traversal over a bounded domain. |
| `recovery` | Interrupt after two rounds, retain a checkpoint, resume without repeating a completed round. |
| `walk_forward` | Keep each test window after its fitting/validation observations, with purge and embargo. |
| `holdout` | Assess final observations only after selection; retain the assessment and its exposure limits. |
| `calibration` | Compare all four calibration/threshold modes and disclose validation reuse. |
| `execution` | Execute recorded observations under explicit costs, product rules and causal clocks. |
| `observed_costs` | Pin recorded execution-cost evidence; compare frozen caller candidates without transferring account fees. |
| `bars` | Form volume, trade and liquidity bars; omit an incomplete final bar. |

Python recovery restores search/results, not arbitrary model weights. Continue inference with the evaluated objects returned in memory, or own the persistence and identity contract yourself.

Walk-forward selection and final assessment remain caller-owned. They do not create guarded project reservations, compiled fold identities or an exposure-history guarantee. Calibration shows an arbitrary fitted classifier through `Sensor.from_python`; its output mapping owns the meaning of the binary decision.

Real TabPFN model validation remains unproved under [#628](https://github.com/Vaquum/Limen/issues/628).
