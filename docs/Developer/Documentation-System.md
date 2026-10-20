# Documentation system contract

Limen's documentation is a research book built around two complete, executable projects: Python Research and Compiled Research. Chapters develop those projects; Reference and Developer provide the detail needed to use and maintain them.

This contract governs the documentation in this repository. The [README](../../README.md) establishes its language and research narrative. [CLAUDE.md](../../CLAUDE.md#research-interface-contract) establishes the research interfaces. Documentation must describe their current behavior and make the evidence behind its examples inspectable.

The [Python project](../examples/python-research/README.md) and
[Compiled project](../examples/compiled-research/README.md) are the canonical
programs governed here. Their entry points, recorded inputs and named scenarios
supply chapter examples; the appendix procedures below maintain their proof.

## Prerequisites

- a Limen checkout and its Python development environment
- the dependencies required by each example being changed
- Node.js 20 or later and `npm ci` in `docs-site`

For the existing source checks, a fresh Python installation is `.[dev,data]`. Research CI uses the hashed dependencies in `requirements/ci/research-env.txt`. Record the interpreter, platform and dependency versions used for execution; a local installation within the package's supported ranges is not the same environment as the Linux CI lock.

## Implementation status

The owned reader and research projects are candidates in this repository. Their execution and browser checks run in `PR Checks Docs Site`. The complete book and released-package publication remain separate delivery obligations; a candidate's successful run does not establish production publication.

| Delivery | Responsibility |
|---|---|
| [Contract](https://github.com/Vaquum/Limen/issues/1051) | Define the book, its example and evidence obligations, and the transition from current documentation. |
| [Reader and projects](https://github.com/Vaquum/Limen/issues/1052) | Build the reader inside Limen, both comprehensive projects, their semantic tests and bindings, and the first complete chapter. |
| [Complete book and publication](https://github.com/Vaquum/Limen/issues/1047) | Complete the maintained corpus and reference coverage, prove released-package reproduction, activate validated publication and retain rollback. |

The current production renderer is Docusaurus. `docs-site/scripts/assemble-docs.mjs` maps maintained sources into Overview, Guides, Reference, Developer and Packages. Existing checks cover source inventory, local links and anchors, fence syntax, selected semantic assertions, public exports, versions, lint, dependency auditing and the static build. `tests/test_python_sensor.py` also executes the marked Python workflow and checks its selection, fitted-state reuse and causal inference.

The candidate first chapter binds exact source regions and retained observations to both projects. The scenario catalogue defines required executions; the evidence checker rejects missing, skipped or stale proof. Selected tests can run independently, but their incomplete evidence cannot publish a reader. The complete suite must execute every required scenario and integrity test before producing successful publication evidence. The reader consumes the tested wheel, input bundle, source identities and current results. Browser checks exercise the real base path and Worker policy, including keyboard use, reduced motion, saved passages, code copying, reference return and mobile navigation. Retained screenshots record the candidate's rendered state.

Integrity tests record native fitting and calibration inputs and forbid fitting after final selection is frozen. The archived CLI commands run in a separate process with the installed wheel. Native result files, checkpoints, round records, audits, final assessment rows and the installed dependency inventory remain available in the evidence archive.

The remaining migration is the difference between the maintained source inventory and `catalogue.json`'s bound pages. Each migrated page must pass its source/reference roles and applicable visible project connections; a repeated link alone cannot establish semantic coverage. Their source footers identify reference excerpts without claiming execution. The first complete chapter is a checkpoint; it does not complete the book. Keep the existing renderer and its source checks until the complete-book publication delivery replaces it.

## Validating the candidate reader

Use Python 3.10–3.13 and Node.js 20 or later. CI uses the policy-pinned Python 3.10 runtime and installs the hash-pinned research environment and the tools in `docs/examples/requirements-quality.txt`. The supplemental documentation type configuration is selected explicitly; it does not shadow the package's strict configuration. Prepare runtime and build wheels before pytest. The proof creates a new environment, installs those wheels offline with hash checking, builds without isolated dependency resolution and installs the candidate wheel there. It imports no dependency from the invoking environment. Linux uses the research CI lock; macOS uses its separately resolved lock with the same version constraints. Both records identify the resolution target and actual interpreter.

From the repository root, after publishing the candidate PR:

| Step | Command |
|---|---|
| 1 | `npm --prefix docs-site ci` |
| 2 | `python docs/examples/check_sources.py` |
| 3 | `python docs/examples/provenance.py --output .docs-evidence/source-provenance.json` |
| 4 | `python docs/examples/prepare_environment.py` |
| 5 | `python -m pytest --import-mode=importlib docs/examples/tests -q -x --tb=short` |

The binding checker uses the reader's locked Markdown parser to identify fenced and indented code blocks, including quoted and tilde forms. Each must occupy the exact location of a validated source or output binding. Install the Node dependencies before running binding tests.

Acquire provenance with full Git history and authenticated read access to GitHub's commit-to-PR associations. Keep its response inventory with the execution evidence. The tests build the actual wheel, install it outside the checkout, unpack the source/input bundle and execute each project's CLI unchanged. They write `.docs-evidence/current.json` as successful only after the required semantic assertions pass. A failed run leaves an explicit running or failed record; it cannot preserve an earlier success.

Then, from `docs-site/`:

| Step | Command |
|---|---|
| 1 | `npm run check` |
| 2 | `npm run security:audit` |
| 3 | `npm run build:reader` |
| 4 | `npx --no-install playwright install chromium` |
| 5 | `npm run check:reader` |

The reader build requires a clean source tree, matching package and current execution evidence. Browser verification checks that evidence and every built file before opening the pages. Use the repository's managed validation runner when it owns the PR; record the necessary commands there and run them in the required order. A background start is not a passing result.

Change canonical project code first, synchronize its marked chapter regions and run the affected proof. Keep displayed observations under their recorded run identity unless their meaning or comparison bounds require a new capture. Never hand-edit a successful current-validation record. Extend the model and scenario catalogue when the public surface changes; a newly exported family cannot silently disappear from the documentation's coverage inventory.

The retained legacy Docusaurus path has two recorded deployment defects:

- `docs-site/src/worker.js` redirects `/robots.txt` to `/limen/robots.txt`, but the build emits no `robots.txt` asset at that destination.
- The Google Fonts import in `docs-site/src/css/custom.css` conflicts with the Worker's CSP: its style and font directives do not admit the required Google origins.

The owned reader self-hosts Archivo/Plex and its browser checks verify loading under the Worker CSP. Production activation must also verify the canonical robots resource. A candidate browser pass does not establish that the legacy production deployment changed.

## Ownership

Limen owns the reader implementation, content, examples, semantic tests, evidence format and publication adapters. Implement them in this repository. The design system is a visual reference. The existing `mikkokotila-com` and `eightchord-com/reader` implementations are references for reading, code and navigation behavior; identify the revision and ownership of any source actually reused.

Keep rendering, product content and research proof separate within Limen. Content supplies identity, routes, chapter order and example bindings. The reader presents them. The execution checks establish what ran and what it returned. Presentation must not manufacture or certify research evidence.

Later reuse across Vaquum products follows the proved Limen implementation. This delivery requires no new shared package, private-repository build dependency or change in the reference repositories. Extraction and other-product adoption are subsequent work.

## Research interfaces

Use the README's names and distinction throughout the book.

| Interface | Research responsibility | Inference handoff |
|---|---|---|
| Python Research | Callers own their data, preparation, models, backtesting and selection in ordinary Python. Custom SFD `params`, `prep` and `model` functions use UEL with caller-controlled search through MSQ. | Wrap evaluated fitted state with `Sensor.from_model` or `Sensor.from_python`; compose selected Sensors with `Cohort.from_python`. |
| Compiled Research | Callers adopt the `Manifest` contract through `MLManifest`, `RuleBasedManifest`, `CompiledSFD` or the CLI. Limen applies the configured pipeline checks and records their evidence. | Reconstruct selected recorded rounds or qualified folds through Trainer and bind compatible Sensors into a Cohort. |

Calling a Manifest from Python remains Compiled Research. UEL, MSQ, indicators, features, targets and scalers are shared machinery; their presence does not identify an interface.

Show both interfaces wherever the subject applies to both. Explain who owns each preparation, evaluation or evidence boundary. Caller-supplied Python declarations do not create compiled provenance. A compiled check does not prove arbitrary feature causality, data correctness or future performance.

An implementation limitation must remain an implementation limitation. Do not rename it methodological inapplicability. If a documented claim exposes a research defect, keep the claim incomplete and route a separately scoped repair; documentation changes must not silently alter the research contract.

## The book and its appendices

The main book carries one research question from recorded observations to selected, reusable Sensors and Cohorts. Guides become part of that explanation. Keep Reference and Developer available for direct lookup and link to them at the point where detail becomes useful.

| Part | Reader task |
|---|---|
| Book | Understand the research decisions, run the projects, inspect their results and continue to the next decision. |
| Reference | Find names, signatures, parameters, defaults, units, outputs, errors, supported combinations and optional dependencies. |
| Developer | Change, test, release, deploy and maintain Limen and the evidence behind the projects. |

Package READMEs retain their ownership, public entry points and adjacent boundaries. Place their navigation with the reference material rather than requiring a separate top-level section.

Maintain one explicit source-to-route map. Every maintained source has exactly one canonical page and route. Generated contents, indexes and search results route to those pages. Optional `navigationTitle` supplies an API lookup label for a narrative page heading; the sidebar, reference index and search share that label from the same map. The authored heading and its anchor remain unchanged. Count the actual corpus: root `README.md`, maintained Markdown under `docs/`, and package READMEs under `limen/`. A fixed page count is not a completeness test.

Every maintained page connects to one or both example projects. The required evidence check now requires the bound-page catalogue and scenario contexts to equal the complete maintained inventory, with no duplicate or opt-out page. Adding a page without those bindings fails before a reader can be built. A research page uses the relevant source, scenario or result. A developer page connects through maintaining, proving or releasing those projects. A contributor-only task does not need an artificial research snippet. Shared applicable capabilities show both interfaces; a tag or repeated boilerplate link alone does not establish coverage.

### Chapter composition

Begin with the decision the reader needs to make. Explain its meaning and consequences in connected prose, then develop the relevant part of the projects. Show what the execution returned, explain the result's limits and lead into the next decision.

Keep prerequisites, inputs, commands, outputs and failure boundaries close to the example they qualify. Link a partial excerpt to its complete project and required earlier state. Use headings for the explanation's structure; do not turn every chapter into the same checklist.

### Reference and developer composition

A reference page names its public surface, import pattern and applicable interfaces before giving signatures, defaults, units, returns and errors. Its concrete examples come from the projects. Preserve depth for registries, Manifest fields, CLI commands, model families and inference contracts.

A developer page states its authority, prerequisites, owned surfaces, executable process, required evidence and failure conditions. Keep build and deployment mechanics in this appendix. They belong in the main book only when they affect a reader's research decision.

## Writing

The README's language must remain coherent across the whole system.

- Lead with current behavior and the reader's decision.
- Use ordinary English and exact domain terms. Prefer concrete verbs and one claim per sentence.
- Keep Python Research and Compiled Research equally explicit. Do not present YAML or the CLI as the universal first path.
- Explain the methodology, observable result and responsibility boundary together.
- Use exact public names, commands, paths, values, units and return fields where they carry meaning.
- Introduce an optional dependency before the first example that needs it.
- Keep measurements tied to their data and environment. Do not turn them into performance promises.
- Distinguish implemented, historical, planned and unproved behavior.
- Author each explanation once; summarize and link from adjacent pages.
- Use relative links between maintained source documents. The root README uses public repository source URLs so its PyPI/package description remains usable; the reader resolves those known source URLs through the same canonical route map.
- Omit praise, intensifiers, ornamental transitions and claims of capability unsupported by source and execution.

The book may be extensive. Each paragraph, table, example and visual must earn its place through the information it carries.

## Two executable projects

Maintain the canonical projects under `docs/examples/python-research/` and `docs/examples/compiled-research/`. They are complete public-interface programs, with documented installation, recorded inputs, entry points and observable outputs. They are not a new research language.

Use named scenarios for choices that cannot coexist in one run. The main path remains readable; scenarios provide the breadth needed for the interface. The coverage catalogue must account for public exports, registries, Manifest fields, CLI commands, supported families and combinations, and the pages explaining them. Preserve the existing public-export, reference and embedded-version checks alongside this catalogue until equivalent protection is demonstrated.

The Python project must show caller-owned preparation, fitting, backtesting, search and selection, followed by reuse of the evaluated fitted state. The Compiled project must show the declared pipeline, validation and execution, recorded results, reconstruction and compatible cohort binding. Include supported recovery, walk-forward, holdout, calibration and execution scenarios where their data and evidence are available. Distinguish caller-owned assessment from guarded project holdout, and ordinary members from qualified recorded folds.

Use recorded data only. Declare each input's origin, SHA-256 digest, relevant time range, row selection and availability clocks. A missing recorded input or optional environment is an unproved boundary, not permission to invent data or output. Real TabPFN model validation remains unproved while the execution tracked in [#628](https://github.com/Vaquum/Limen/issues/628) is absent; a skipped test cannot support a successful-run claim.

### Acquisition and execution

Publish a version-pinned bundle containing the projects and recorded inputs. Document the matching released package, extras, working directory and exact entry-point commands. Examples and fixtures are not assumed to be inside the installed wheel.

Tests run those commands and sources unchanged. Accept recorded-input paths as documented arguments or resolve them from a documented working directory. Do not rewrite a manifest, replace a model or monkeypatch a data loader to create an undisclosed test version. Assertions and observation of fitting are permitted; substituting different research behavior is not.

Prove the installation outside the source checkout. Verify the imported distribution and location, package version, input digests, path behavior and meaningful results. Candidate checks execute the candidate wheel. Production validation executes the actual released wheel; an editable installation plus a version-availability lookup does not establish that result.

`docs/examples/released_wheel.py` downloads the exact version from public PyPI metadata and retains the wheel, metadata and digest receipt. The research tests accept `--released-wheel RECEIPT` to install that wheel instead of building a candidate. Both paths use the same isolated environment, unchanged project commands and semantic assertions. A released wheel must match the checkout's version and every shipped Python source file. Selecting or downloading a wheel alone does not prove execution; publication still requires the complete passing suite and merged-source evidence.

## Examples and their bindings

Classify every Python, YAML, JSON and Bash fence. Its role determines the proof it needs.

| Role | Obligation |
|---|---|
| Executable project excerpt | Exact named region of canonical project source, with scenario, prerequisites and executing semantic tests. |
| API signature | Current public source and signature; clearly presented as reference, not a runnable workflow. |
| Package layout | Exact tracked package inventory, derived from its package README location and presented as text. No executable-workflow claim. |
| Partial fragment | Explicit earlier state and full-project route; no claim that the fragment runs alone. If it comes from an executed project, retain that binding. |
| Command | Current arguments and dependency/working-directory context; execute commands used to establish a workflow claim. |
| Observed output | Retained bytes from an identified successful execution and its declared comparison rules. |

Keep standalone Python parseable; identify non-standalone fragments explicitly. Preserve plain language info strings when adding role markers: `python`, `python-fragment`, `yaml`, `json`, `bash`, `sh`, `text` and unlabelled display fences must be accounted for. Put binding metadata outside the fence, or update its parser and migration in the same change. `python-fragment` remains a partial fragment; `bash` and `sh` carry commands; `text` may carry observed output or non-executable reference text. Language alone does not distinguish signatures from executable Python. Existing fences remain under the current checks until the book migration classifies the complete corpus. Informational fences do not escape source checking or establish execution. Manifest-based examples instantiate `MLManifest` or `RuleBasedManifest`; base `Manifest` is an abstract interface.

Package signatures and constant definitions are derived from the current Python syntax tree. Signature bindings retain the declared parameters, defaults, annotations and decorators while replacing executable bodies with an ellipsis. Definition bindings retain the actual module assignment. Their source hashes and merged-PR identities are checked separately from the projects; the current wheel proof also compares every shipped Python source file with the tested tree. These blocks describe package interfaces and do not claim to execute a research workflow.

Package layouts use `layout:package` with their own `limen/**/README.md` as source. Their text must equal the immediate files and directories in Git's tracked package inventory. Adding, removing or renaming a tracked member requires updating that diagram. Module-responsibility notes remain authored reference text beside the inventory.

Each bound page records its relevant scenario identities in `catalogue.json` under `page_contexts`. Its visible Markdown links must reach every applicable canonical project, unless the page is that project's own README. The reader's parser checks those links as well as the fences; a link inside an HTML comment cannot establish the connection. Reference and layout roles do not replace the scenario context or assert that every alternative helper ran.

Authored and displayed executable excerpts must equal their canonical source regions. Tests execute that source, compare the excerpts and reject a missing region, duplicate identity, wrong interface, stale binding or changed display. Test deliberate divergence so the guarantee itself has evidence.

Assertions must establish the promised behavior. Depending on the claim, this includes chronological separation, fitting and selection cutoffs, comparable evaluation rows, independent cost/timing calculations, retained fitted state, reconstruction, availability, causal prefixes and meaningful failure behavior. Successful process exit alone is insufficient.

Retained output displayed in the book must match its evidence record exactly. Retain a named, immutable output record for display; each fresh execution compares its semantic results with that record under declared per-metric tolerances and deterministic settings. In-tolerance numerical drift alone does not require editing displayed output. A changed claim, failed invariant or exceeded tolerance requires inspection and, where warranted, a regenerated record and synchronized prose. Record timestamps, attempt identities and timing measurements as run evidence; do not require fresh executions to reproduce a previous run's incidental identifiers. Explain why a reader's environment can produce different digits and what invariant must still hold.

Verify a retained observation's source region against its recorded execution commit. Verify the current excerpt against the current executable source separately. A later source edit must pass fresh semantic comparison; it must not relabel the earlier observation as a new run.

## Source and current validation

Every displayed example exposes its immutable source identity and separate current validation. A source hash does not prove compatibility with a later package.

| Record | Required identity |
|---|---|
| Source | Source file and region, content digest, immutable commit, and the latest merged PR changing that file. |
| Validation | Executed source tree, package version and wheel digest, interpreter/platform/dependency identity, recorded-input digests, scenario and test identities, assertions, observed output and comparison rules. |
| Publication | Merged release/tree, successful validation identity, retained bundle digests and the deployed build. |

Limen owns the versioned schema and checker under `docs/examples/`. The reader consumes their presentation data. It must not invent a merged PR, infer a passed test from a code fence or attach an old successful result to a changed package.

Resolve a production file’s latest changing commit along the first-parent history of the published main revision, comparing each commit with its first parent. Associate that commit with the merged PR whose base is this repository’s main branch. Branch synchronization merges do not establish publication provenance. A file with no verifiable merged changing PR has unavailable provenance and cannot receive a successful production footer.

Generate source footers from actual Git and merged-PR history during the build. Do not commit a source file's own final commit hash into that file. A candidate preview identifies unmerged source explicitly. Candidate evidence records the PR head, tested merge commit and executed tree separately; the checkout used by a pull-request workflow may be the test merge rather than the head. Production records the published main commit/tag and actual released wheel digest. Production requires the real merged changing PR and passing validation for the package and tree being published.

Keep the source commit/PR visible with each example. Make the complete project, input, environment, test and retained output evidence reachable from it without an authenticated browser session.

## Changing examples or research behavior

A change to project code, preparation, numerical behavior or displayed results requires the corresponding documentation, bindings and semantic proof to change together.

1. Identify affected projects, scenarios, source regions, pages and results in the coverage catalogue.
2. Include their paths in the slice's Surfaces before implementation.
3. Change the canonical source and synchronized excerpts; regenerate observed results through execution.
4. Run the required proof in its declared environment and inspect the changed assertions and results.
5. Publish only evidence for the exact candidate or release being judged.
6. Follow the repository's version, CHANGELOG and citation requirements even when the visible change is documentation.

Binding tests must request current-run execution evidence themselves. Running one test in isolation cannot depend on another test having already populated a file. Comparing authored output with a stale committed output alone does not establish freshness.

## Build and evidence flow

The existing required `PR Checks Docs Site` context will own the comprehensive example proof and reader checks. Extend its job in this order:

1. Identify the checked-out revision and source tree.
2. Install the policy-pinned Python 3.10 runtime and the hashed research dependencies; build the candidate wheel and install it in the isolated execution environment.
3. Execute the canonical projects and semantic/binding tests under `docs/examples/tests/`, producing evidence for those exact inputs.
4. Run Node's source, route, lint, dependency and reader checks against that same-job evidence.
5. Build and browser-check the site, retaining the execution and browser artifacts.

Missing, failed, skipped required proof or wrong-revision evidence fails the check. Do not pass evidence through an independent workflow that can race the build. Record the dependency lock digest and its resolution target as well as the actual interpreter patch version, platform and runner image. Installing a lock resolved for Python 3.10 under Python 3.12 does not make it a Python 3.12 resolution. Include the browser runner, Chromium and system dependencies in the locked/provisioned toolchain. Audit the complete shipped dependency graph through Limen's lockfile.

The book tests provide documentation evidence. They do not replace a public research slice’s ordinary `tests/` proof required by the Interface Contract. Documentation maintenance may cite the required Docs Site check where a governance declaration needs a check reference.

Keep comprehensive docs execution outside default `tests.run` discovery. The package's required test, coverage and 240-second runtime obligations remain unchanged. The measured suite executes on Python 3.12; separate runtime and coverage jobs use Python 3.10 to read its artifacts. The initial `fb5dc4f2` baseline measured 252.659 seconds against 240 seconds. Isolating new docs work does not repair or waive that failure, nor establish the health of a later revision; judge the actual candidate's checks.

## Visual and motion contract

The selected authority is the experience style guide at `http://127.0.0.1:4328/`, inspected from reference revision `f683f91`. It supplies the visual language, not an implementation destination or build dependency. The reader delivery must commit screenshots and a token, typography, geometry and motion snapshot in Limen so the local server and mutable reference branch are not continuing dependencies.

### Color and type

| Role | Value |
|---|---|
| Navy / raised navy | `#142038` / `#1B2A49` |
| Stock / sheet | `#F4F2EC` / `#FBFAF6` |
| White / ink | `#F7F6F2` / `#0E1A2B` |
| Muted / silver | `#5E6977` / `#AAB3BD` |
| Gold / gold ink | `#DFB967` / `#805A1B` |
| Syntax blue / green | `#AFCBE3` / `#D1DCC6` |
| Profile labels | `#8F9AA7` |

Self-host Archivo Variable and IBM Plex Mono under the site's CSP. Display uses Archivo weight 200, width 125%, `.04em` tracking and `.95` line height. Prose uses Archivo weight 400, normal width, `17px` with `1.6` line height and a reading measure up to `60ch`. Code uses Plex Mono weight 400, `14px / 22px`, exact indentation and disabled ligatures. Labels use Plex Mono at `11px`, weight 500 and `.14em` tracking.

The composition uses a centered field up to `1320px`, `20px` side gutters, an `8px` base unit, fine rules and square corners. A five-to-seven split places explanation beside evidence where the content calls for it. Stack the split at the reference's `1024px` boundary and adapt the inspector at `640px`; prove the book's actual navigation and overflow at those sizes.

The fixed navy/paper composition replaces the old light/dark/system theme-switch requirement. Test the same composition under both OS color preferences and forced colors. Preserve readable text, syntax, selection and visible focus rather than relying on a theme preference to supply contrast.

### Motion and reading

Motion must explain continuity between the selected object, its detail and the reader's return. Keep values and identities stable through the transition. Use subtle parallax only where it makes that relationship clearer. Long-form prose and code remain stable while being read.

The reference uses one damped spring with stiffness 180 and damping 20; retargeting preserves position and velocity. Its selected entry/return timings are 2400/1600 ms and local transitions use 220 ms. Apply these values to the corresponding interaction, not indiscriminately to every control. Wheel, touch and navigation input interrupt camera travel. Scrolling alone must not open an unrelated continuation.

Reduced motion reaches the same content, selection and focus directly. Keyboard controls expose the same information as pointer controls. The reference's visible focus is `2px` with a `3px` offset. Static prose, source and navigation remain readable without JavaScript.

## Navigation and saved position

The reader must support sequential reading, direct reference lookup, search and movement between applicable Python and Compiled explanations. Keep global parts, chapter contents and current passage distinct. Source edit links target real maintained files.

A reference excursion preserves its originating chapter, passage, interface and keyboard focus. Returning must restore those, not merely reopen the chapter at its top. Browser history and explicit return controls must agree.

Saved reading position is local and scoped by product and content identity. Prove save/reload, isolation between products on one origin, and changed or removed passages. A stale position must not silently restore into unrelated text. Deep links retain their own target. Notes, annotations, accounts and reader analytics are outside this delivery.

Code controls preserve exact source when copying, exclude line-number gutters, retain indentation and expose language, full source and provenance. Long code and tables must remain usable on a phone without horizontal page overflow.

## Routes, security and discovery

During this prelaunch restructuring, routes and anchors may change deliberately. Blanket redirects are not required under the operator’s explicit decision for this restructuring. The publication delivery must enumerate links already shipped in README/package metadata and record their retained, redirected or deliberately retired destinations; absolute links to the documentation origin belong in the same route-map checks. This contract slice changes no routes. Migrate internal links and ensure source links, anchors, navigation, search, canonical metadata and sitemap all use the same map. Once a route is intentionally retained, it must resolve to its declared content.

Maintain one canonical origin and product base path. Limen owns `/limen/`; shared-host routing, root resources, robots and sitemap ownership must be explicit. A product adapter must not take over another product's prefix. Prove status codes, redirects and errors through the real deployment adapter.

Preserve HSTS, CSP, permissions, referrer, content-type and frame protections. Build metadata and local search from the canonical content map. Verify every public route, sitemap/robots agreement, source edit target and emitted asset. Check external destinations separately from local links; do not claim that local link validation establishes external availability.

## Publication and rollback

The existing Cloudflare Git-integrated build runs independently of GitHub checks. Keep it through the reader checkpoint. Before merging the first removal of the production renderer, prepare and activate its replacement, record the actual cutover and disable the independent deployment path. Required credentials or dashboard actions must be resolved at that concrete deployment boundary.

Production publication follows successful package publication. Resolve the merged release and available wheel, obtain the pinned example/input bundle, and execute the documented commands against that wheel outside the source checkout. Build the site from the resulting exact-release evidence. A push to `main` alone is not publication approval.

Assemble evidence and attach bundles through deterministic code, independently of release-note prose. Retain content-addressed source, input, output, validation and site bundles as public GitHub Release assets and static evidence. Actions artifacts may supplement diagnostics but are not sufficient for anonymous durable access. Verify bundle identities rather than assuming a filename is immutable.

`docs/examples/publication.py` seals an already tested reader only when every maintained page is bound, the checkout is clean at its merged release tag, every source has merged-PR provenance, and the executed wheel matches its retained PyPI receipt. The execution record stays unchanged. Production adds separate hashes for that record, the project/input bundle and the site payload. The payload excludes evidence and build metadata to avoid hashing itself. The rollback archive retains the site and evidence; its manifest and the rollback archive itself remain detached. Restore both alongside the extracted site. `verify_publication.py` retrieves the advertised artifacts without credentials, checks their hashes and source identities against the public GitHub Release, compares the wheel with current public PyPI metadata, and checks the active home page against the sealed payload. These tools do not themselves activate a deployment.

Prove anonymous HTTP retrieval and digests for the advertised source, results, validation and rollback artifacts. Exercise pending or failed proof, wrong revisions, PR-versus-merged identities and missing or failed package publication. These conditions must retain the previous validated deployment. A unit test of a JSON record alone cannot prove ordering between release, execution and deployment.

Retain the previous validated site and its evidence through cutover and post-deployment smoke checks. Verify the candidate preview before activation and the deployed routes, headers, assets and proof afterward. Remove the replaced build path after acceptance; do not leave permanent competing production renderers.

## Verification and completion

For the current renderer, run the existing commands after publishing the useful first commit and non-draft PR, following repository workflow:

| Step | Command |
|---|---|
| 1 | `python -m pytest tests/test_docs_surface.py -q` |
| 2 | `npm --prefix docs-site run security:audit` |
| 3 | `npm --prefix docs-site run check` |

These commands prove their current scope. They do not establish the future reader, comprehensive examples or evidence-bound production.

The reader delivery adds executing project/binding tests under `docs/examples/tests/` and browser/build checks under `docs-site/`. The final delivery applies them to the complete corpus and actual publication. Keep every existing required package, security, typing, coverage, runtime, install and packaging check effective.

The acceptance matrix covers:

- both complete projects, supported scenarios, documented acquisition and meaningful results
- every maintained page and applicable public capability, retaining reference depth
- exact source/display/output bindings and deliberate divergence failures
- merged-source identity and separate current package/environment/input/test evidence
- desktop and phone, both OS color preferences, forced colors, keyboard and reduced motion
- readable prose, source and navigation without JavaScript
- local search, exact code copy, reference return and persisted-position safety
- representative Book, Reference, Developer and package pages against the retained visual reference
- declared static asset and runtime budgets, measured on the real reader and enforced against regression
- real publication ordering, anonymous evidence retrieval and retained rollback

Record the first complete chapter's commit, preview URL and evidence bundle. It must show both research interfaces, real output/provenance, sequential navigation and a reference excursion with return. Keep the parent delivery open until the complete book, reference/developer corpus and publication meet the same contract.

## Read next

- [Docs hub](../README.md)
- [Developer home](README.md)
- [Writing docstrings](Writing-Docstrings.md)
- [Packaging](Packaging.md)
- [Release policy](Release-Policy.md)
- [Security assurance case](Security-Assurance-Case.md)
