# Compiled Research project

The Manifest owns preparation and its configured checks. Trainer reconstructs selected recorded rounds and validates their metrics.

The [research chapter](../../Research-Book.md) develops the project. The [documentation contract](../../Developer/Documentation-System.md) defines the proof required before this candidate is published as the complete book.

## Run the selection scenario

Install the matching `vaquum-limen[data]` distribution. From the example bundle root, supply the directory containing the recorded input files:

```bash
python compiled-research/research.py --scenario selection --data-dir inputs --output-dir results/compiled
```

In a repository checkout, the corresponding recorded inputs are in `tests/fixtures/`; set `--data-dir` to that directory and keep generated output outside the source tree. The command checks the file digest before research. It emits `summary.json`; the Compiled project also retains its run definition, results and reconstruction artifacts.

## Scope

The first scenario establishes selection and inference handoff. The reader slice is still in development: its required catalogue, remaining scenarios, fresh-wheel reproduction and browser acceptance must pass before this project is described as comprehensive. Real TabPFN validation remains unproved under [#628](https://github.com/Vaquum/Limen/issues/628).
