# Python Research project

Python functions define preparation, fitting, backtesting, and selection. The selected Sensors reuse the evaluated fitted objects.

The [research chapter](../../Research-Book.md) develops the project. The [documentation contract](../../Developer/Documentation-System.md) defines the source, execution and publication proof maintained with the book.

The search scenario also compares baseline and compressed preparation on 1,440 recorded hourly bars. It retains each prepared train/validation/test matrix, a fixed-order fractional feature, ternary forward-close labels, training-fitted robust statistics, an explicit feature drop and optional two-component PCA. Independent tests reconstruct the labels, differenced values, scaler statistics, PCA subspace and Ridge forecasts. Python owns separate partition preparation. The Compiled YAML variant uses the native pipeline's admitted CCO context and its own declared date population. These are preparation checks, not a feature-importance ranking or profitability claim.

## Acquire the matching project

Download [acquire.sh](https://docs.vaquum.fi/limen/source/docs/examples/acquire.sh). It reads one public publication manifest, verifies the exact project-bundle and wheel digests, extracts the recorded inputs and installs the matching wheel with `data`, `stats` and `boosting` extras into a new environment. It requires Python 3.10–3.13, `curl`, `jq`, `tar` and OpenSSL. Run the Python entry point below from the acquired bundle.

<!-- example: acquire-python-project source=docs/examples/commands.json region=acquire-project scenario=selection interface=python role=executed -->
```bash
sh acquire.sh limen-research
```

Enter `limen-research/` and activate its environment with `. .venv/bin/activate`. Run every following command from that bundle root. Keep `publication.json`; it pins the selected release and artifact digests even after the public site advances. The examples and inputs are separate from the installed wheel. To repeat a past run, retain that manifest, bundle, wheel and environment record.

## Source encodings

`source_cases.py` reads the same 1,440 recorded hourly observations from the bundled CSV, ZIP and Arrow files, compares latest-row semantics and retains their schemas/timestamps/values for independent checks. Their original Parquet source remains pinned. [Historical data reference](../../Historical-Data-Reference.md#recorded-format-equivalence) gives both interfaces and the format constraints.

## Preparation diagnostics

`helper_cases.py` applies every exported indicator and feature to recorded bars or trades, retains each output, fits every registered scaler on the training rows and transforms later rows without changing its state. The transforms are separate training-sample diagnostics. `target_cases.py` exercises every ordinary target constructor/transform on bounded recorded populations; it retains the volatility sanity gate and raw-bar refusals for targets requiring supplied outcome columns. Separate supplied-column cases demonstrate those contracts on the bundle’s derived labels. Line geometry, regime and microstructure helpers receive their declared context; forward breakout labels remain diagnostic labels and do not enter model inputs. The tests independently verify time identity, ROC, moving average, robust scaling and transform arithmetic. These diagnostics accompany the research model; they are not hundreds of simultaneous model inputs.

## Prerequisites

The project uses Python 3.10–3.13 and the matching wheel and input bundle identified by its execution evidence. Selection needs the `data` extra. The full project needs `vaquum-limen[data,stats,boosting]`; Sobol uses statistics, and residual correction uses XGBoost.

## Run the selection scenario

Install the matching `vaquum-limen[data]` distribution. From the example bundle root, supply the directory containing the recorded input files:

<!-- example: python-selection-command source=docs/examples/commands.json region=python-selection scenario=selection interface=python role=executed -->
```bash
python python-research/research.py --scenario selection --data-dir inputs --output-dir results/python
```

From a repository checkout, work in `docs/examples/`, use `--data-dir ../../tests/fixtures` and put generated output under `../../.docs-evidence/`. The command checks the recorded file digest before research. It writes `research.csv` and `summary.json`. The evaluated models and Sensors remain in caller-owned memory during the run; those two files do not restore fitted state. Import `run_research` to continue using the returned Cohort and retained models in the same Python process.

## Continue through the research decisions

The [linked-learning case](../../Linked-Models.md) fits caller-owned Ridge and XGBoost residual stages, selects correction depth on validation, and retains each component's output. The recovery scenario also executes the [captured-state case](../../Run-And-Recover.md), preserving explicitly retained fitted pipelines in a trusted `.uel` snapshot and checking their restored forecasts.

Search also executes all eight [common strategy baselines](../../Common-Strategies.md) on the recorded 2024 hourly source. Each comparison names its own assessment window. The Python case owns next-open-to-close one-bar trades with a declared 2 bp round-trip cost; the Compiled case uses the native lagged held-position snapshot with 10 bp fee and 5 bp slippage per fill. Independent assertions check causal signals and each policy's arithmetic; these are different trading interpretations, not a cross-engine return ranking.

`--scenario all` runs the named scenarios in separate output directories. `--scenario` also accepts one name from the table. Use a new output directory for a fresh run; the recovery scenario creates its interrupted run and resumes it internally.

The execution scenario compares cash spot with a linear perpetual using the recorded funding history, realized rates and mark prices. The shared `recorded_inputs.py` checks and normalizes those records; the native funding ledger exposes each payment. Both runs retain their execution contract. Tests require signal fills to follow its one-hour delay and compare every displayed fill with the native Parquet record. The bar scenario also forms three CFR diagnostic bars from the pinned 90-second trade slice. This is shared preprocessing on recorded events, not a claim that the short slice can support model selection.

Search also fits 30 window/penalty configurations for recorded RMSE correlation and held-out surrogate parameter importance. Four further runs exercise metric-driven sanity, saturation, focus and correlation controls. Their observations, audit entries and checkpoint states remain separate. Search also runs a separate resource-controlled grid. `BudgetReducer(max_permutations=3)` reduces a four-candidate budget to three completed rounds through the shared feedback controller. Its audit and checkpoint retain the intervention and remaining queue state. A separate mixed run combines that reducer with a callback and a file-driven injection. The proof checks ordering, deduplication, resulting domain and independently computed scores. These controls change the observations collected; they do not establish better selection.

Traversal, preparation and recovery use the first 1,440 recorded hourly observations. The Python common-baseline case uses all 8,784 recorded hourly bars and its 1,598-bar assessment partition. The Compiled YAML has separate explicit calendar-date partitions. Recovery compares the completed resumed run with a separate uninterrupted run. Walk-forward uses the first 384 hourly observations; its inference callback declares their one-hour availability interval. Selection, ordinary final assessment and classifier calibration retain their quarter-hour population.

The observed-cost scenario creates one immutable calibration from the recorded Chainticks population and pins its digest separately. Four caller-owned fits compare original and calibrated ledgers across two hours and both binary and signed exposure. Prices are recorded midpoints encoded as points; capital and strategy exposure are software scenarios. The calibration archive postdates evaluation, historical delivery is unknown and sell support is concentrated. These are retrospective stress results. The bundle retains the source attribution, CC BY 4.0 license and reproduction files. The [Backtest reference](../../Backtest.md#frozen-candidates-in-custom-python-research) uses the exact same model source.

| Scenario | Decision and output |
|---|---|
| `search` | Compare traversal, preparation, eight common rules, residual learning, parameter analysis, budget reduction, and mixed feedback under their declared populations. |
| `recovery` | Compare interrupted/resumed execution with a complete run; separately capture and restore explicitly retained fitted pipelines. |
| `walk_forward` | Keep each test window after its fitting/validation observations, with purge and embargo. |
| `holdout` | Assess final observations only after selection; retain the assessment and its exposure limits. |
| `calibration` | Compare all four calibration/threshold modes and disclose validation reuse. |
| `execution` | Execute recorded observations under explicit costs, product rules and causal clocks. |
| `observed_costs` | Pin recorded execution-cost evidence; compare frozen caller candidates without transferring account fees. |
| `bars` | Form fixed activity bars and adaptive trade-point bars, retaining complete constituents and omitted trailing observations. |

Checkpoint recovery preserves search and results. The separate captured-state case restores explicitly retained fitted pipelines from a trusted snapshot. In-process inference uses the evaluated objects returned by the selection functions.

Walk-forward selection and final assessment remain caller-owned. They do not create guarded project reservations, compiled fold identities or an exposure-history guarantee. Calibration shows an arbitrary fitted classifier through `Sensor.from_python`; its output mapping owns the meaning of the binary decision.

The search variant also fits a caller-owned three-class nearest-neighbor model. Training-only next-return quantiles define its labels; a later partition supplies macro precision, recall, AUC and accuracy. The [metric reference](../../Standard-Metrics-Library.md#recorded-three-class-evaluation) shows the exact function and current results.

The [optional model-template campaign](../compiled-research/README.md#optional-model-campaign) records a real TabPFN CPU run separately. Python Research keeps the choice and fitting of an ordinary Python model caller-owned.

The [adaptive bar cases](../../Data-Bars.md#recorded-trade-points-exercise-adaptive-formation) use a pinned unique-clock view of the recorded trade archive. They retain their actual formed populations and the native fold-promotion limitation.
