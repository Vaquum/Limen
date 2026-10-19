# Python Research project

Your functions own preparation, fitting, backtesting and selection. The selected Sensors reuse the evaluated fitted objects.

The [research chapter](../../Research-Book.md) develops the project. The [documentation contract](../../Developer/Documentation-System.md) defines the proof required before this candidate is published as the complete book.

## Run the selection scenario

Install the matching `vaquum-limen[data]` distribution. From the example bundle root, supply the directory containing the recorded input files:

```bash
python python-research/research.py --scenario selection --data-dir inputs --output-dir results/python
```

From a repository checkout, work in `docs/examples/`, use `--data-dir ../../tests/fixtures` and put generated output under `../../.docs-evidence/`. The command checks the recorded file digest before research. It writes `research.csv` and `summary.json`. The evaluated models and Sensors remain in caller-owned memory during the run; those two files do not restore fitted state. Import `run_research` to continue using the returned Cohort and retained models in the same Python process.

## Scope

The first scenario establishes selection and inference handoff. The reader slice is still in development: its required catalogue, remaining scenarios, fresh-wheel reproduction and browser acceptance must pass before this project is described as comprehensive. Real TabPFN validation remains unproved under [#628](https://github.com/Vaquum/Limen/issues/628).
