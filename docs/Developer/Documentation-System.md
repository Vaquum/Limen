# Documentation system contract

Limen's documentation is a research book built around two complete, executable projects\. Python Research uses Python functions; Compiled Research uses YAML and CLI commands\. Chapters develop those projects; Reference and Developer provide the detail needed to use and maintain them\.

This contract governs the documentation in this repository. The [README](../../README.md) establishes its language and research narrative. [Repository law](../../.github/REPOSITORY-LAW.md) establishes the delivery laws. Documentation must describe current behavior and make the evidence behind its examples inspectable.

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

The owned reader and research projects are published together with release-bound execution evidence. Their execution and browser checks run in `PR Checks Docs Site`. The public publication manifest identifies the active release; a successful candidate run does not establish production publication.

| Delivery | Responsibility |
|---|---|
| [Contract](https://github.com/Vaquum/Limen/issues/1051) | Define the book, its example and evidence obligations, and the transition from current documentation. |
| [Reader and projects](https://github.com/Vaquum/Limen/issues/1052) | Build the reader inside Limen, both comprehensive projects, their semantic tests and bindings, and the first complete chapter. |
| [Complete book and publication](https://github.com/Vaquum/Limen/issues/1047) | Complete the maintained corpus and reference coverage, prove released-package reproduction, activate validated publication and retain rollback. |

The owned reader is the sole renderer in this checkout. `docs-site/reader/content.json` maps every maintained source into Book, Reference and Developer. Existing checks cover source inventory, local links and anchors, fence syntax, selected semantic assertions, public exports, versions, lint, dependency auditing and the evidence-consuming static build. The production Worker serves the release named in its public publication manifest. `tests/test_python_sensor.py` also executes the marked Python workflow and checks its selection, fitted-state reuse and causal inference.

The maintained corpus binds exact source regions and retained observations to both projects. The scenario catalogue defines required executions; the evidence checker rejects missing, skipped or stale proof. Selected tests can run independently, but their incomplete evidence cannot publish a reader. The complete suite must execute every required scenario and integrity test before producing successful publication evidence. The reader consumes the tested wheel, input bundle, source identities and current results. Browser checks exercise the real base path and Worker policy, including keyboard use, reduced motion, saved passages, code copying, reference return and mobile navigation. Retained screenshots record the candidate's rendered state.

Integrity tests record native fitting and calibration inputs and forbid fitting after final selection is frozen. The archived CLI commands run in a separate process with the installed wheel. Native result files, checkpoints, round records, audits, final assessment rows and the installed dependency inventory remain available in the evidence archive.

The maintained source inventory equals `catalogue.json`'s bound pages. Each page must pass its source/reference roles and applicable visible project connections; a repeated link alone cannot establish semantic coverage. Their source footers identify reference excerpts without claiming execution. Source validation and production publication are separate outcomes. Every publication must pass the current released-wheel, preview and activation checks below.

## Validate the reader and research

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
| 3 | `npx --no-install playwright install chromium webkit` |
| 4 | `npm run check:reader` |

The reader build requires a clean source tree, matching package and current execution evidence. Browser verification checks that evidence and every built file before opening the pages. Use the repository's managed validation runner when it owns the PR; record the necessary commands there and run them in the required order. A background start is not a passing result.

Change canonical project code first, synchronize its marked chapter regions and run the affected proof. Keep displayed observations under their recorded run identity unless their meaning or comparison bounds require a new capture. Never hand-edit a successful current-validation record. Extend the model and scenario catalogue when the public surface changes; a newly exported family cannot silently disappear from the documentation's coverage inventory.

The owned reader emits the canonical robots resource and self-hosts Archivo/Plex under the Worker CSP. Both are checked before publication and over anonymous HTTP before activation. A source merge does not activate a new Worker version; publication verifies preview and production independently.

## Ownership

Limen owns the reader implementation, content, examples, semantic tests, evidence format and publication adapters. Implement them in this repository. The design system is a visual reference. The existing `mikkokotila-com` and `eightchord-com/reader` implementations are references for reading, code and navigation behavior; identify the revision and ownership of any source actually reused.

Keep rendering, product content and research proof separate within Limen. Content supplies identity, routes, chapter order and example bindings. The reader presents them. The execution checks establish what ran and what it returned. Presentation must not manufacture or certify research evidence.

Later reuse across Vaquum products follows the proved Limen implementation. This delivery requires no new shared package, private-repository build dependency or change in the reference repositories. Extraction and other-product adoption are subsequent work.

## Research interfaces

Use the README's names and distinction throughout the book.

| Interface | Research responsibility | Inference handoff |
|---|---|---|
| Python Research | Callers own their data, preparation, models, backtesting and selection in ordinary Python. Custom SFD `params`, `prep` and `model` functions use UEL with caller-controlled search through MSQ. | Wrap evaluated fitted state with `Sensor.from_model` or `Sensor.from_python`; compose selected Sensors with `Cohort.from_python`. |
| Compiled Research | Callers declare the complete research definition in YAML and execute it through the CLI. Limen applies the configured pipeline checks and records their evidence. | `limen assess` reconstructs validation-selected recorded rounds or qualified folds and binds their Sensors into a Cohort. |

Compiled Research uses YAML declarations and CLI commands throughout. Its canonical directory contains no Python drivers. Tests may inspect artifacts in Python, but every execution claim must run the unchanged YAML through the installed CLI. UEL, MSQ, indicators, features, targets and scalers are shared machinery; their presence does not identify an interface.

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
- Keep Python Research and Compiled Research equally explicit. Python Research uses ordinary Python; Compiled Research uses YAML and CLI throughout.
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

The Python project must show caller-owned preparation, fitting, backtesting, search and selection, followed by reuse of evaluated fitted state. The Compiled project must show YAML declaration, CLI validation and execution, recorded results, CLI reconstruction and compatible cohort binding. Its complete canonical manifest covers composable features; separate YAML variants cover incompatible choices such as target families, fixed and walk-forward splits, bar formation and guarded final assessment.

Use recorded data only. Declare each input's origin, SHA-256 digest, relevant time range, row selection and availability clocks. A missing recorded input or optional environment is an unproved boundary, not permission to invent data or output. The optional TabPFN campaign retains a separately provisioned real-model observation with strict current-source and numerical checks; a skipped test cannot support a successful-run claim.

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

Keep standalone Python parseable; identify non-standalone fragments explicitly. Preserve plain language info strings when adding role markers: `python`, `python-fragment`, `yaml`, `json`, `bash`, `sh`, `text` and unlabelled display fences must be accounted for. Put binding metadata outside the fence, or update its parser and migration in the same change. `python-fragment` remains a partial fragment; `bash` and `sh` carry commands; `text` may carry observed output or non-executable reference text. Language alone does not distinguish signatures from executable Python. Every maintained fence remains classified and source-bound under the current checks. Informational fences do not escape source checking or establish execution. Manifest-based examples instantiate `MLManifest` or `RuleBasedManifest`; base `Manifest` is an abstract interface.

Package signatures and constant definitions are derived from the current Python syntax tree. Signature bindings retain the declared parameters, defaults, annotations and decorators while replacing executable bodies with an ellipsis. Definition bindings retain the actual module assignment. Their source hashes and merged-PR identities are checked separately from the projects; the current wheel proof also compares every shipped Python source file with the tested tree. These blocks describe package interfaces and do not claim to execute a research workflow.

Package layouts use `layout:package` with their own `limen/**/README.md` as source. Their text must equal the immediate files and directories in Git's tracked package inventory. Adding, removing or renaming a tracked member requires updating that diagram. Module-responsibility notes remain authored reference text beside the inventory.

Each bound page records its relevant scenario identities in `catalogue.json` under `page_contexts`. Its visible Markdown links must reach every applicable canonical project, unless the page is that project's own README. The reader's parser checks those links as well as the fences; a link inside an HTML comment cannot establish the connection. Reference and layout roles do not replace the scenario context or assert that every alternative helper ran.

Authored and displayed executable excerpts must equal their canonical source regions. Tests execute that source, compare the excerpts and reject a missing region, duplicate identity, wrong interface, stale binding or changed display. Test deliberate divergence so the guarantee itself has evidence.

Assertions must establish the promised behavior. Depending on the claim, this includes chronological separation, fitting and selection cutoffs, comparable evaluation rows, independent cost/timing calculations, retained fitted state, reconstruction, availability, causal prefixes and meaningful failure behavior. Successful process exit alone is insufficient.

Retained output displayed in the book must match its evidence record exactly. Retain a named, immutable output record for display; each fresh execution compares its semantic results with that record under declared per-metric tolerances and deterministic settings. In-tolerance numerical drift alone does not require editing displayed output. A changed claim, failed invariant or exceeded tolerance requires inspection and, where warranted, a regenerated record and synchronized prose. Record timestamps, attempt identities and timing measurements as run evidence; do not require fresh executions to reproduce a previous run's incidental identifiers. Explain why a reader's environment can produce different digits and what invariant must still hold.

Verify a retained observation's source region against its recorded execution commit. Verify the current excerpt against the current executable source separately. A later source edit must pass fresh semantic comparison; it must not relabel the earlier observation as a new run.

A `result` marker selects fields from a current, hash-verified scenario output. Array field projections render retained rows as a table; authors do not copy numerical values into prose. Missing fields, malformed projections, changed result bytes and missing source identities fail before publication. Historical `example-output` blocks keep their original observation identity separately.

## Source and current validation

Every displayed example exposes its immutable source identity and separate current validation. A source hash does not prove compatibility with a later package.

| Record | Required identity |
|---|---|
| Source | Source file and region, content digest, immutable commit, and the latest merged PR changing that file. |
| Validation | Executed source tree, package version and wheel digest, interpreter/platform/dependency identity, recorded-input digests, scenario and test identities, assertions, observed output and comparison rules. |
| Publication | Merged release/tree, successful validation identity, retained bundle digests and the deployed build. |

Limen owns the versioned schema and checker under `docs/examples/`. The reader consumes their presentation data. It must not invent a merged PR, infer a passed test from a code fence or attach an old successful result to a changed package.

Resolve a production file’s latest changing commit along the first-parent history of the published main revision, comparing each commit with its first parent. Associate that commit with the merged PR whose base is this repository’s main branch. Branch synchronization merges do not establish publication provenance. A file with no verifiable merged changing PR has unavailable provenance and cannot receive a successful production footer.

For a retained original PR head, use its latest shared ancestor on main’s first-parent history. Merging the PR does not turn its side-branch commits into mainline commits; its changed sources remain candidate evidence until the merged revision is tested. The provenance command accepts `--root PATH` to inspect another full-history checkout without changing that checkout’s source or revision; `--output` retains its path relative to the invoking directory.

Classify source bytes against both that mainline boundary and the ordinary common ancestor. Restoring a file to its pre-PR bytes remains a candidate change when it differs from the already-merged branch ancestor. The original evidence guard continues to reject changed source presented as merged.

Retained checkouts also need the corrected checker when their original test harness and reader validate evidence. From that checkout’s root, run its declared interpreter with the reviewed `docs/examples/retained_checker.py`, followed by the original provenance command, `-m pytest` arguments or evidence-checker command. The launcher imports its reviewed provenance dependency normally before loading the retained harness. The original product, test assertions, evidence validator and isolated wheel execution keep their own source files. Each invocation retains the product head/tree, checker file digests, interpreter, arguments and actual exit code; this tooling receipt certifies no research result by itself.

Install a fresh executable with that interpreter and `retained_checker.py --install-launcher PATH`. Set the reader’s existing `LIMEN_PYTHON` to that executable so its original evidence-checker command uses the same dependency. Required CI exercises provenance acquisition, the full wheel/project proof and the reader build through this route. A source-only `provenance.py --root` invocation does not select the checker for those later stages. Caller admission and the retained owner’s actual qualification remain separate obligations; old failed runs retain their outcomes.

The retained-checker regressions run first in the same required Docs Site job, under their own 60-second bound. Each complete Python/CLI entry command has a 120-second limit. The enlarged proof has a 240-second outer limit for two isolated installations, both campaigns and all semantic/integrity assertions; package-suite runtime remains governed separately by `tests/runtime_budget.json`. Every existing project assertion remains required. Checker tests revalidate captured source metadata against the actual retained Git tree; their fixture setup does not regenerate that metadata on every run. Raised failures retain their nonzero exit codes and tracebacks in addition to the invocation receipt.

Generate source footers from actual Git and merged-PR history during the build. Do not commit a source file's own final commit hash into that file. A candidate preview identifies unmerged source explicitly. Candidate evidence records the PR head, tested merge commit and executed tree separately; the checkout used by a pull-request workflow may be the test merge rather than the head. Production records the published main commit/tag and actual released wheel digest. Production requires the real merged changing PR and passing validation for the package and tree being published.

Keep the source commit/PR visible with each example. Make the complete project, input, environment, test and retained output evidence reachable from it without an authenticated browser session.

## Changing examples or research behavior

A change to project code, preparation, numerical behavior or displayed results requires the corresponding documentation, bindings and semantic proof to change together.

1. Identify affected projects, scenarios, source regions, pages and results in the coverage catalogue.
2. Record their paths in the delivering issue before implementation.
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

The book tests provide documentation evidence. They do not replace a public research change’s ordinary `tests/` proof. Documentation maintenance may cite the required Docs Site check as its evidence.

Keep comprehensive docs execution outside default `tests.run` discovery. The package's required test and coverage obligations remain unchanged; `tests/runtime_budget.json` is the current runtime authority. The measured suite executes on Python 3.12; separate runtime and coverage jobs use Python 3.10 to read its artifacts. Judge the actual revision’s package and documentation checks separately; neither waives the other.

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

Saved reading position is local and scoped by product and content identity. Reload and browser history restore the relevant entry; an ordinary page link opens at the destination's beginning. Prove save/reload, Back/Forward, isolation between products on one origin, and changed or removed passages. A stale position must not silently restore into unrelated text. Deep links retain their own target, revealing its interface when needed. Notes, annotations, accounts and reader analytics are outside this delivery.

Code controls preserve exact source when copying, exclude line-number gutters, retain indentation and expose language, full source and provenance. Long code and tables must remain usable on a phone without horizontal page overflow.

## Routes, security and discovery

During this prelaunch restructuring, routes and anchors may change deliberately. Blanket redirects are not required under the operator’s explicit decision for this restructuring. The publication delivery must enumerate links already shipped in README/package metadata and record their retained, redirected or deliberately retired destinations; absolute links to the documentation origin belong in the same route-map checks. The retained `docs-site/reader/route-decisions.json` inventories every route and README/package link in the pre-cutover source revision. It records the retained or deliberately retired URL and its current reading destination. Source checks verify the pinned baseline and complete inventory; browser checks prove retained and retired status codes and replacement destinations. The build seals this record as `route-decisions.json`. Migrate internal links and ensure source links, anchors, navigation, search, canonical metadata and sitemap all use the same map. Once a route is intentionally retained, it must resolve to its declared content.

Maintain one canonical origin and product base path. Limen owns `/limen/`; shared-host routing, root resources, robots and sitemap ownership must be explicit. A product adapter must not take over another product's prefix. Prove status codes, redirects and errors through the real deployment adapter.

Preserve HSTS, CSP, permissions, referrer, content-type and frame protections. Build metadata and local search from the canonical content map. Verify every public route, sitemap/robots agreement, source edit target and emitted asset. Check external destinations separately from local links; do not claim that local link validation establishes external availability.

## Publication and rollback

The owned reader uses one release-bound publisher. Its independent Cloudflare Git build remains disconnected. A successful candidate proves the PR’s source and examples; publication additionally requires the actual released wheel, immutable public artifacts, an inactive preview and verified activation or rollback.

Production publication follows successful package publication. Resolve the merged release and available wheel, obtain the pinned example/input bundle, and execute the documented commands against that wheel outside the source checkout. Build the site from the resulting exact-release evidence. A push to `main` alone is not publication approval.

Assemble evidence and attach bundles through deterministic code, independently of release-note prose. Retain content-addressed source, input, output, validation and site bundles as public GitHub Release assets and static evidence. Actions artifacts may supplement diagnostics but are not sufficient for anonymous durable access. Verify bundle identities rather than assuming a filename is immutable.

Keep a wheel's original distribution, version and compatibility filename so a downloaded artifact remains installable. Its static parent directory carries its SHA-256; the Release copy must have identical bytes. Keep each detached manifest as `publication-SHA256.json` on the Release. The site's `publication.json` selects its manifest; a rebuild must preserve every earlier manifest and artifact so the previous deployment and rollback remain verifiable. Never replace a Release asset with different bytes.

`docs/examples/publication.py` seals an already tested reader only when every maintained page is bound, the checkout is clean at its merged release tag, every source has merged-PR provenance, and the executed wheel matches its retained PyPI receipt. The execution record stays unchanged. Production adds separate hashes for that record, the project/input bundle and the site payload. The payload excludes evidence and build metadata to avoid hashing itself. The rollback archive retains the site and evidence; its manifest and the rollback archive itself remain detached. Restore both alongside the extracted site. `verify_publication.py` retrieves the advertised artifacts without credentials, checks their hashes and source identities against the public GitHub Release, compares the wheel with current public PyPI metadata, and checks every served document and asset against the sealed payload. The Worker and build share one security-header declaration. The sealed site retains that declaration as `security-headers.json`; verification compares all six values on every response with that release's policy, including the 404 response. An unknown route must return the sealed 404 with status 404. These tools do not themselves activate a deployment.

`.github/workflows/docs_publish.yml` accepts only a successful canonical package-publishing run and its run-scoped identity artifact. `release_identity.py` verifies that run, merged tag and exact wheel. `release_assets.py` uploads immutable artifacts without replacement and downloads the actual public project/input bundle. The full proof executes that downloaded bundle and wheel outside the checkout. A release superseded while queued cannot activate.

`deploy_publication.py` requires disconnected Git build triggers before staging. It reads the production version before and after capturing rollback hashes and rejects a changed version before uploading. It then uploads an inactive Worker version and verifies its anonymous Version URL. Activation rechecks disconnected triggers, preview evidence and the recorded production version. It checks the latest package release again immediately before deployment, after deployment and after production verification; a release superseded during those operations cannot receive acceptance. A failed activation or production check restores the recorded predecessor and verifies its home-page bytes and, when available, its complete release envelope. Each activation and rollback allows five seconds for global deployment propagation, confirms the active version again, and applies the same exact HTTP verification. A legacy predecessor is identified by its Worker version and home-page digest; it cannot claim the release envelope used by later reader publications. Live acceptance remains necessary. Setup and credential requirements are in [the deployment instructions](../../docs-site/CLOUDFLARE.md).

The publisher pins Wrangler in its own lockfile and requires Node.js 22 or later. The required Docs Site job runs the research reader under Node.js 20, then installs and audits the separate publisher graph. Both audits inspect the complete graph and allow no advisory exceptions. Temporary HTTP 502, 503 or 504 responses from the public npm audit endpoint receive at most three attempts, with every response retained. Advisories, authentication failures and malformed reports fail without retries; exhausted gateway attempts also fail. Contributor and fork builds do not require deployment credentials.

Prove anonymous HTTP retrieval and digests for the advertised source, results, validation and rollback artifacts. Exercise pending or failed proof, wrong revisions, PR-versus-merged identities and missing or failed package publication. These conditions must retain the previous validated deployment. A unit test of a JSON record alone cannot prove ordering between release, execution and deployment.

Retain the previous validated site and its evidence through cutover and post-deployment smoke checks. Verify the candidate preview before activation and the deployed routes, headers, assets and proof afterward. Preserve the preceding Worker version and its immutable rollback identity until production acceptance finishes. Keep one production publisher.

## Verification and completion

For the current renderer, run the existing commands after publishing the useful first commit and non-draft PR, following repository workflow:

| Step | Command |
|---|---|
| 1 | `python -m pytest tests/test_docs_surface.py -q` |
| 2 | `npm --prefix docs-site run security:audit` |
| 3 | `npm --prefix docs-site run check` |

These commands prove source integrity, dependency security and the owned reader build. The build requires the complete same-head project proof above; local commands do not establish a production deployment.

The reader delivery supplies executing project/binding tests under `docs/examples/tests/` and browser/build checks under `docs-site/`. The final delivery applies them to the complete corpus and actual publication. Keep every existing required package, security, typing, coverage, runtime, install and packaging check effective.

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

## Optional model evidence

The [TabPFN project command](../examples/compiled-research/README.md#optional-model-campaign) uses a real checkpoint pinned by upstream revision and SHA-256. Its ordinary test verifies the retained observation against every current package Python file, the package version, exact YAML/shell source and recorded input, then independently reconstructs the labels, accuracy and pairwise AUC. A changed file or version fails admission; editing the JSON is not a replacement for model execution.

To regenerate, provision a separate supported environment with the matching non-editable Limen wheel and its `data,tabpfn` extras. Use TabPFN 6.4.1 and the recorded environment as the reproduction baseline. Unpack the exact canonical bundle in a fresh directory, then invoke `docs/examples/optional_models.py` with `--root` pointing to the clean tested checkout, `--project` to that bundle root and `--output docs/examples/retained/tabpfn.json`. Put the matching environment's `limen` first on `PATH`. The capture command verifies that the installed package is outside the checkout and byte-identical to its Python source, runs the unchanged shell/YAML command, validates the genuine outputs and writes a new timestamped observation. Commit that observation with the affected sources and rerun the ordinary documentation proof.

The recorded run is CPU, one ensemble, Python 3.12 and macOS ARM64. Its source/version equality check is current validation of that observation; ordinary CI does not download and run the optional model again. Other devices, dependency environments and stochastic replay require their own execution evidence. Keep that distinction visible wherever this result is cited.

## Prose editing

The contributor guide [Edit and verify the research book](Editing-The-Book.md) develops reader, editor, and source-authoring tasks. This section owns deployment configuration, session behavior, and activation requirements.

Authoring is configured separately from the released book at `limen-editor.vaquum.fi/limen/edit/`. Activation remains tracked in [issue #1207](https://github.com/Vaquum/Limen/issues/1207). GitHub sign-in requires current write permission on Vaquum/Limen. Read the branch and revision shown there before editing; the public book may identify an earlier package release.

Edit existing words in place, then use Review to inspect drafts across pages. Commit sends those drafts to an origin branch. Further commits accumulate on that branch. Open PR includes the committed changes and starts the normal checks and review. Local Undo does not revert a remote commit. Code examples, recorded results and structural changes remain source work.

The implementation is configured by `docs-site/editor/wrangler.json`. Build authoring assets with `npm run build:editor` in `docs-site`; deploy a clean reviewed revision with the locked Wrangler in `docs-site/publisher`. Git build triggers remain disconnected. Staging and production use separate Workers, domains, Durable Object storage and secrets. Neither deployment activates or replaces the released public book.

The GitHub App needs Metadata read and Contents, Pull requests and Issues write for Limen only; it needs no Workflows permission. Configure `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` and an independent base64-encoded 32-byte `SESSION_KEY` as Worker secrets in each environment. The callback is the corresponding authoring origin followed by `/limen/edit/auth/callback`. Installation is an explicit maintainer authorization. Until installation and remote acceptance are recorded in issue #1207, this source configuration does not establish a live editor.

Sessions are server-owned and expire after one hour. Sign-out invalidates the session before requesting GitHub token revocation. Rotation of the session key invalidates existing sessions. Drafts belong to the signed-in writer and delivery, and persist only in that browser. Commit or download pending prose before explicitly discarding it at sign-out. A moved source, missing branch or conflicting PR requires reconciliation; the editor never force-pushes or merges on the writer's behalf.
