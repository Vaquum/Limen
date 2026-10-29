# Third-Party Notices

Limen's Python and JavaScript dependencies are declared in:

- `pyproject.toml`
- `docs-site/package.json`
- `docs-site/package-lock.json`
- `docs-site/publisher/package.json`
- `docs-site/publisher/package-lock.json`

Dependency license review is required before release when dependencies are added, removed, or materially upgraded.

## Version 6.0.0 dependency review

The release review on 29 October 2026 covers the dependency changes since 5.46.2 and the separately executed TabPFN environment. Third-party dependency code and checkpoint weights remain external to the Limen wheel. Upstream distributions retain their own bundled-library notices.

| Reviewed dependency | Release use | Reviewed license surface |
|---|---|---|
| `cloudpickle` 3.1.2 | Runtime captured-object serialization | BSD-3-Clause |
| `hmmlearn` 0.3.3 | Optional HMM backend | BSD-3-Clause, with the upstream copyright notice |
| `torch` 2.13.0 and 2.14.1 | Locked neural proof and separate TabPFN proof | Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0, and MIT notices in the reviewed wheels |
| `tabpfn` 6.4.1 | Optional classifier and its recorded CPU example | Prior Labs License 1.2, derived from Apache-2.0 with an additional attribution provision |
| Pinned TabPFN-v2 checkpoint at `f851f2a3c941544733b712d8c0f96dfae9b28862` | Separately downloaded optional model | Prior Labs License 1.1 with its additional attribution provision |

Built with PriorLabs-TabPFN: the optional TabPFN example uses [Prior Labs TabPFN](https://github.com/PriorLabs/TabPFN). The unmodified [package license](docs/examples/compiled-research/licenses/TabPFN-code-6.4.1-LICENSE.txt) and [pinned checkpoint license](docs/examples/compiled-research/licenses/TabPFN-v2-model-LICENSE.txt) are retained with that project and in the distribution's license data. Their additional attribution and distributed-model naming terms remain applicable to downstream use. TabPFN is a separate optional license surface, rather than Apache-2.0-only dependency metadata.

Pandas moves from runtime requirements to test/development requirements. The existing boosting, data, statistics, indicator-comparison, reader, typography, and publisher dependencies retain their separately declared distributions and notices. This review records the named package versions and checkpoint revision; later dependency changes require their own review.

## Current Known Notes

- Reader and publisher audits inspect their complete locked dependency graphs and permit no advisory exceptions.
- Optional `tabpfn` support is not installed by default; install it only when the TabPFN workflow is required.
- Optional `hmmlearn` support is installed through the `hmm` extra for GaussianHMM return models.
- The `tabpfn` extra must be reviewed as a separate optional license surface before release because its upstream package metadata can include attribution terms beyond a plain Apache-2.0-only summary.
- `typing_extensions` (PSF-2.0, pure Python, no transitive dependencies) is a runtime dependency for `@override` on Python 3.10; when the Python floor reaches 3.12 the import moves to `typing` and the dependency is removed.

## Research reader typography

The owned reader self-hosts unmodified Archivo Variable and IBM Plex Mono font binaries from `@fontsource-variable/archivo` 5.3.0 and `@fontsource/ibm-plex-mono` 5.3.0. Their SIL Open Font License texts are copied with the fonts into the candidate build. The original font names and license notices are retained. Limen’s reader implementation is authored in this repository; the local mikkokotila-com and Eightchord readers informed its reading, code and navigation behavior and are not build dependencies.

The deployment toolchain pins Cloudflare Wrangler 4.151.2 (MIT OR Apache-2.0) in a separate lockfile. Its `workerd` runtime uses Apache-2.0; the complete transitive license inventory remains in that lockfile. This toolchain deploys the owned reader and is not a dependency of the Python package or the reader's browser code.

## Recorded execution-cost evidence

`tests/fixtures/observed_costs/` contains modified data from Chainticks/perp-data, revision `54b07b4bebc5574a065d3fbf4432634cf9148f50`, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Chainticks Perp Data, derived from public chain/archive state. https://chainticks.com

Modified: normalized/hashed, selected recorded populations and separately labelled diagnostics. Deterministic hashes normalize account identity; they do not anonymize public addresses. The full publisher notice, source pins, attribution and reproduction instructions are retained in the fixture's `license.json`, `provenance.json` and `reproduce.py`. This data license is separate from the package MIT license.
