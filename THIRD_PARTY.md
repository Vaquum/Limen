# Third-Party Notices

Limen depends on Python and JavaScript open source packages declared in:

- `pyproject.toml`
- `docs-site/package.json`
- `docs-site/package-lock.json`
- `docs-site/publisher/package.json`
- `docs-site/publisher/package-lock.json`

Dependency license review is required before release when dependencies are added, removed, or materially upgraded.

## Current Known Notes

- Reader and publisher audits inspect their complete locked dependency graphs and permit no advisory exceptions.
- Optional `tabpfn` support is not installed by default; install it only when the TabPFN workflow is required.
- The `tabpfn` extra must be reviewed as a separate optional license surface before release because its upstream package metadata can include attribution terms beyond a plain Apache-2.0-only summary.
- `typing_extensions` (PSF-2.0, pure Python, no transitive dependencies) is a runtime dependency for `@override` on Python 3.10; when the Python floor reaches 3.12 the import moves to `typing` and the dependency is removed.

## Research reader typography

The owned reader self-hosts unmodified Archivo Variable and IBM Plex Mono font binaries from `@fontsource-variable/archivo` 5.3.0 and `@fontsource/ibm-plex-mono` 5.3.0. Their SIL Open Font License texts are copied with the fonts into the candidate build. The original font names and license notices are retained. Limen’s reader implementation is authored in this repository; the local mikkokotila-com and Eightchord readers informed its reading, code and navigation behavior and are not build dependencies.

The deployment toolchain pins Cloudflare Wrangler 4.151.2 (MIT OR Apache-2.0) in a separate lockfile. Its `workerd` runtime uses Apache-2.0; the complete transitive license inventory remains in that lockfile. This toolchain deploys the owned reader and is not a dependency of the Python package or the reader's browser code.

## Recorded execution-cost evidence

`tests/fixtures/observed_costs/` contains modified data from Chainticks/perp-data, revision `54b07b4bebc5574a065d3fbf4432634cf9148f50`, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Chainticks Perp Data, derived from public chain/archive state. https://chainticks.com

Modified: normalized/hashed, selected recorded populations and separately labelled diagnostics. Deterministic hashes normalize account identity; they do not anonymize public addresses. The full publisher notice, source pins, attribution and reproduction instructions are retained in the fixture's `license.json`, `provenance.json` and `reproduce.py`. This data license is separate from the package MIT license.

- Optional `hmmlearn` support is installed through the `hmm` extra for GaussianHMM return models.
