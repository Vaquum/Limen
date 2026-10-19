# Compiled Research project

The Manifest owns preparation and its configured checks. Trainer reconstructs selected recorded rounds and validates their metrics.

The [research chapter](../../Research-Book.md) develops the project. The [documentation contract](../../Developer/Documentation-System.md) defines the proof required before this candidate is published as the complete book.

## Run the selection scenario

Install the matching `vaquum-limen[data]` distribution. From the example bundle root, supply the directory containing the recorded input files:

```bash
python compiled-research/research.py --scenario selection --data-dir inputs --output-dir results/compiled
```

From a repository checkout, work in `docs/examples/`, use `--data-dir ../../tests/fixtures` and put generated output under `../../.docs-evidence/`. The command checks the recorded file digest before research. It writes `summary.json`, `results.csv`, `metadata.json` and the compressed round artifacts. Trainer reconstructs the selected round from those artifacts and validates the replayed metrics; it does not restore saved model weights.

## Scope

The first scenario establishes selection and inference handoff. The reader slice is still in development: its required catalogue, remaining scenarios, fresh-wheel reproduction and browser acceptance must pass before this project is described as comprehensive. Real TabPFN validation remains unproved under [#628](https://github.com/Vaquum/Limen/issues/628).
