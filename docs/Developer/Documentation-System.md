# Documentation system contract

This page defines the documentation product proven in Limen and the contract for reproducing it in other production repositories. It separates the portable system from Limen-specific content so repositories share standards, composition, behavior, and visual language without inheriting Limen claims.

`MUST`, `SHOULD`, and `MAY` identify required, recommended, and optional parts of the portable contract. A requirement listed as a current gap is not yet a Limen guarantee and blocks extraction of the shared system.

## Prerequisites

- a Limen repository checkout with the existing project `.venv`
- Python development and data dependencies for the semantic documentation gate
- Node.js 20 or later
- docs-site dependencies installed from the committed lockfile with `npm ci`

The equivalent fresh Python installation is `.[dev,data]`. CI installs the hashed research environment from `requirements/ci/research-env.txt` and then installs Limen without dependency resolution.

## Quality bar

Documentation is release-ready only when it is:

- correct against current source, exports, templates, workflows, and runtime behavior
- complete across the maintained public Markdown corpus
- coherent from product entry through workflows, reference, maintenance, and package boundaries
- runnable in the dependency environment each example declares
- consistent in terminology, units, page roles, links, composition, and visual treatment
- accessible and usable across keyboard, desktop, mobile, light, and dark contexts
- secure and discoverable through correct headers, canonical metadata, sitemap, and robots behavior
- mechanically protected in proportion to the claim through tests, lint, link checking, and the rendered-site build

## Portability boundary

The reference has two layers.

| Layer | Owns | Must not own |
|---|---|---|
| Shared documentation system | information architecture, page composition, source-to-route invariants, visual tokens, interaction behavior, quality gates, parity evidence, rollout, and drift control | product names, product claims, product routes, repository coordinates, package inventories, risk wording, or narrative sequence |
| Product profile | product identity, tagline, repository URL, site origin and base path, source inventory and routes, canonical terminology, product narrative, dependencies, examples, risk boundary, and ownership | shared rendering logic, shared CSS behavior, shared acceptance rules, or duplicated copies of the shared system |

The current Limen implementation is the reference behavior, not yet a reusable package boundary. Product-specific values remain distributed across `product-docs.json`, `docusaurus.config.js`, `assemble-docs.mjs`, `custom.css`, `worker.js`, and `wrangler.jsonc`. Extraction happens only after Limen passes the parity protocol and the remaining gaps in this page are closed.

A shared package or preset is portable only when:

- its configuration schema can express every product-specific value
- its fixtures contain no `Limen`, `Vaquum`, `Vaquum/Limen`, `/limen/`, or Limen source-path literals
- a product can upgrade the shared version without copying or editing shared files
- product profiles cannot disable required gates without an approved, time-bounded exception
- one build path remains after migration; the old path is removed after rollback expiry

## Limen reference profile

This profile is the frozen interpretation of the current Limen case. Other repositories replace these values; they do not generalize them into the shared layer.

| Property | Limen value |
|---|---|
| Product | `Limen` |
| Product repository | `https://github.com/Vaquum/Limen` |
| Site origin | `https://docs.vaquum.fi` |
| Base path | `/limen/` |
| Edit branch | `main` |
| Maintained Markdown sources | 69: root product home, 42 files under `/docs`, and 26 package `README.md` files under `/limen` |
| Rendered sitemap URLs | 76: 69 maintained pages, 6 generated category indexes, and search |
| Top-level sections | Overview, Guides, Reference, Developer, Packages |
| Site generator | Docusaurus 3, locked by `docs-site/package-lock.json` |
| Hosting | Cloudflare Worker with static assets |
| Local search | `@easyops-cn/docusaurus-search-local` |
| Theme | Vaquum light/dark design in `docs-site/src/css/custom.css` |
| Source-to-route authority | `docs-site/scripts/assemble-docs.mjs` |

These counts are acceptance baselines, not universal targets. A repository may have a different corpus size while preserving the same ownership model, five-section architecture, page contracts, rendering behavior, and gates.

## Source ownership

- [README.md](../../README.md) is the product home and first-success path.
- [docs/README.md](../README.md) is the public task router.
- `/docs` owns public concepts, workflows, guides, and reference.
- `/docs/Developer` owns contributor and maintainer guidance.
- `README.md` files under `/limen` explain package ownership, boundaries, and entry points; they route to canonical docs rather than duplicating them.
- `docs-site/scripts/assemble-docs.mjs` is the complete source-to-route map for the hosted site.
- `.generated`, `.docusaurus`, and `build` are derived outputs and MUST NOT be hand-authored or committed.

Author a claim once whenever practical. Secondary pages summarize and link to the canonical explanation.

## Information architecture

The site presents five top-level sections:

| Section | Responsibility |
|---|---|
| Overview | product boundary, system story, task routing, and roadmap |
| Guides | end-to-end jobs and operational workflows |
| Reference | interfaces, schemas, defaults, outputs, and edge cases |
| Developer | contribution, documentation, release, packaging, security, and maintenance |
| Packages | module ownership, public entry points, and nested package boundaries |

Top-level categories are collapsed by default. Nested package READMEs live under a collapsed Internal packages group so full corpus coverage does not flatten the sidebar.

Every maintained source MUST map to exactly one destination and stable public route. Every route MUST have one primary page role. Category indexes are generated navigation surfaces, not alternate canonical explanations.

## Narrative spine

Major Limen pages agree on this product-specific sequence:

1. `HistoricalData` or an external compatible frame supplies Bitcoin market data.
2. Optional data bars, indicators, features, transforms, scalers, and targets define the research surface.
3. A YAML manifest is the default operator-facing experiment definition.
4. The CLI validates, profiles, dry-runs, and runs that manifest.
5. Universal Experiment Loop is the engine beneath CLI execution and the direct Python extension surface.
6. Log, benchmark, and backtest surfaces explain experiment outcomes.
7. Trainer replays selected rounds with their original manifest configuration, validates metrics, and returns Sensors.
8. Cohort binds selected Sensors into a multi-member inference surface.
9. Trade decisioning and execution occur outside Limen.

Every adopting repository MUST define its own sequence in its product profile. Shared templates may require a narrative spine but MUST NOT supply Limen's sequence.

## Page composition

Composition is the required order of information, not a demand for identical headings. A page may omit a non-applicable element only when the omission is explicit.

### Product home

1. product identity and one-sentence value
2. owned and excluded product boundary
3. current capabilities
4. minimum install and first successful workflow
5. observable outputs or artifacts
6. risk boundary
7. routes by reader task
8. contribution, support, security, citation, and license

### Docs hub

1. product in one page
2. start routes grouped by reader job
3. product narrative sequence
4. complete top-level docs map
5. owned and excluded product boundary
6. explicit next routes

### Guide

1. job, outcome, and current scope
2. prerequisites or an explicit statement that none are required
3. ordered procedure with at least one concrete command or example
4. expected output, artifact, or observable result
5. failure boundaries and material edge cases
6. next task

### Reference

1. covered surface and abstraction boundary
2. public names and import pattern
3. signatures, parameters, defaults, returns, units, and side effects
4. minimum concrete example
5. edge cases, errors, and optional dependencies
6. relationship to adjacent surfaces
7. next reference or workflow

### Developer page

1. purpose and authority
2. prerequisites
3. scope and ownership boundaries
4. executable process or checklist
5. required proof and failure conditions
6. review or maintenance boundary
7. related maintainer routes

### Package README

1. package path and one-sentence responsibility
2. canonical public docs
3. what the package owns and does not own
4. source-true public entry points
5. adjacent packages and optional dependencies
6. compact source-tree orientation when it materially helps navigation
7. operational caveats
8. next routes

## Visual system

`docs-site/src/css/custom.css` is the visual authority. Adopting repositories MUST consume the same shared tokens and component rules; product-specific branding enters through an explicit profile or documented token override.

### Color tokens

| Token | Light | Dark | Role |
|---|---|---|---|
| `--vaquum-paper` | `#F8F8F8` | `#121212` | page and navigation background |
| `--vaquum-paper-2` | `#F3F3F3` | `#231F20` | active, code, and secondary surface |
| `--vaquum-ink-deep` | `#121212` | `#F8F8F8` | headings and strongest text |
| `--vaquum-ink` | `#231F20` | `#F3F3F3` | body and link text |
| `--vaquum-ink-soft` | `#444444` | `#D3D3D3` | secondary text |
| `--vaquum-ink-mute` | `#808080` | `#808080` | tertiary labels |
| `--vaquum-rule` | `#D3D3D3` | `#444444` | dividers and borders |
| `--vaquum-accent` | `#DC65A6` | `#EAA3C8` | active and hover state |
| `--vaquum-coral` | `#F16068` | `#F16068` | code strings |
| `--vaquum-lime` | `#DDD941` | `#DDD941` | success |
| `--vaquum-cyan` | `#C4E8F4` | `#C4E8F4` | information and code constants |

### Typography and geometry

| Element | Contract |
|---|---|
| Body | IBM Plex Sans intent, `17px`, `1.55` line height |
| Monospace | IBM Plex Mono intent, `14.5px`, `1.65` line height |
| Reading column | `680px` maximum |
| H1 | `44px`, weight `600`, `1.1` line height, maximum `28ch` |
| Intro after H1 | `19px`, maximum `52ch` |
| H2 | `13px`, weight `600`, uppercase, `0.08em` tracking, top rule |
| H3 and H4 | `13px`, weight `600`, uppercase, `0.08em` tracking |
| Navigation and sidebar | `13px` |
| Breadcrumb and pagination label | `11px`, uppercase, `0.14em` tracking |
| Navbar | `56px` high |
| Desktop sidebar | `320px` wide |
| Corners and shadows | zero radius, no shadow |
| Code block | secondary surface with a `3px` accent left rule |

The responsive breakpoint is `996px`. Below it, desktop sidebars disappear, the navigation toggle appears, content padding becomes `24px 16px 64px`, and tables scroll horizontally rather than forcing viewport overflow.

### Interaction contract

- Navbar order is Home, Overview, Guides, Reference, Developer, Packages, then GitHub.
- Light, dark, and system theme choices are supported.
- Search is local, keyboard reachable, and links to a complete search page.
- Top-level categories start collapsed; the current section expands.
- The desktop docs sidebar is hideable.
- Mobile uses a navigation drawer and collapsible on-page table of contents.
- Breadcrumbs, heading anchors, previous/next pagination, copy-code controls, and a skip-to-content link remain available.
- Every page's edit link targets its real source under repository `/edit/main/`.
- Blog and standalone pages are disabled; documentation owns the root route.

The IBM Plex faces are the intended typography. The current production CSP does not allow the external Google Fonts import or font origin, so a system fallback may render in production. This is a known pre-extraction defect, not an allowed design variation.

## Writing rules

- Lead with current behavior and reader impact.
- Give each page one primary role and one primary audience task.
- Prefer exact commands, paths, values, units, signatures, return fields, and observable results over abstractions.
- Use American English for shared prose: `artifact`, `behavior`, `optimization`.
- Use canonical component capitalization: `Single-File Decoder`, `Reference Architecture`, `HistoricalData`, `Universal Experiment Loop`.
- Use `python` fences only for standalone parseable code. Use `python-fragment` for fluent-chain fragments or partial code.
- State required extras before the first example that imports an optional dependency.
- Do not present local measurements as stable API guarantees.
- Do not describe planned, historical, or external behavior as current product behavior.
- Use relative links between maintained source documents so assembly can rewrite them.
- End task-oriented pages with an explicit next route.

## Source-backed claims

Use the narrowest authoritative source:

| Claim | Limen authority |
|---|---|
| import or export | package `__init__.py` and an import smoke test |
| callable arguments or defaults | current function or class signature |
| YAML field or template | schema, rules, compiler, and bundled template |
| reducer or scaler name | current registry |
| result field or artifact | implementation and focused test |
| package dependency | `pyproject.toml` and locked CI environment |
| release behavior | current workflow and script |
| hosted route | assembler map and Docusaurus build |
| visual value | `docs-site/src/css/custom.css` |
| deploy behavior or header | `worker.js`, `wrangler.jsonc`, and a production HTTP response |

When prose and source disagree, fix the prose or explicitly route a separate behavior defect. Documentation work MUST NOT silently change runtime contracts.

## Examples

Examples satisfy the level they imply:

- syntax examples parse
- import examples import in the declared extras environment
- command examples use current argument order and names
- runnable workflows complete against a bounded fixture or isolated environment
- output examples contain only fields the implementation can produce in that mode
- copied examples state destructive, costly, networked, or long-running effects before execution

Base `Manifest` is an abstract interface whose `prepare_data()` raises `NotImplementedError`; runnable examples instantiate `MLManifest` or `RuleBasedManifest`.

## Links and route policy

- Local links and fragments resolve in source and in the assembled site.
- Public links use the canonical route shape without an optional trailing slash.
- Every assembled page receives a `custom_edit_url` targeting its real source.
- Repository files not mapped as docs may link to GitHub; maintained documentation is assembled instead of silently falling back to a blob link.
- Published routes are stable API. A changed or removed route requires an explicit redirect, link migration, sitemap update, and parity exception.
- External links require automated status checking before cross-repository rollout; current Limen gates validate local links and anchors only.

## Assembly and configuration

| Surface | Responsibility | Portability status |
|---|---|---|
| `docs-site/product-docs.json` | product identity, tagline, version label, base path, section names, repository URL | partial product profile |
| `docs-site/scripts/assemble-docs.mjs` | complete source map, routes, category metadata, front matter, link rewriting | shared algorithm with Limen-specific map and URLs still embedded |
| `docs-site/docusaurus.config.js` | generator, metadata, search, navigation, footer, syntax highlighting, theme loading | shared configuration with some Limen and Vaquum literals |
| `docs-site/sidebars.js` | autogenerated navigation | shared |
| `docs-site/src/css/custom.css` | visual tokens, typography, components, responsive behavior | candidate shared theme |
| `docs-site/package.json` and lockfile | commands and deterministic JavaScript dependency graph | candidate shared toolchain |
| `docs-site/src/worker.js` | base-path routing and security headers | shared behavior with Limen path embedded |
| `docs-site/wrangler.jsonc` | Cloudflare project, assets, domain, compatibility date | product deployment profile |
| `tests/test_docs_surface.py` | source inventory, semantics, links, anchors, examples, versions, exports | shared test framework with Limen assertions |
| `.github/workflows/pr_checks_docs_site.yml` | dependency install, audit, and required site check | candidate shared CI |

`assemble-docs.mjs`:

1. removes and recreates the generated docs directory
2. maps every maintained source to one destination and route
3. writes source-aware front matter and edit URLs
4. rewrites relative documentation links to assembled destinations
5. preserves repository links only for non-doc files
6. normalizes source Markdown for MDX
7. creates collapsed category metadata

The source inventory test fails if a maintained Markdown source is added without a route or if the map references a missing source.

## Delivery, security, and discovery

The production contract includes:

- one canonical origin and normalized base path
- canonical and language-alternate metadata for every rendered page
- Open Graph and Twitter summary metadata from the product profile
- a sitemap containing every public page and generated category index
- root and bare-base redirects to the canonical product path
- HSTS, CSP, permissions, referrer, content-type, and frame headers on assets, redirects, and errors
- a `404` response outside the owned base path
- a valid robots resource that points crawlers to the canonical sitemap

The current Worker redirects `/robots.txt` to `/limen/robots.txt`, but the built site does not provide that target. The resulting `404` is a known pre-extraction defect.

## Verification matrix

| Concern | Current Limen mechanism | Status before extraction |
|---|---|---|
| maintained source inventory and one-to-one routing | `tests/test_docs_surface.py` | automated |
| source-linked semantic assertions | `tests/test_docs_surface.py` | automated |
| local links and anchors | `tests/test_docs_surface.py` plus Docusaurus broken-link failure | automated |
| Python, YAML, JSON, and Bash fence syntax | `tests/test_docs_surface.py` | automated |
| release and embedded version consistency | `tests/test_docs_surface.py` | automated |
| Markdown style | `npm run lint` | automated |
| assembly and production build | `npm run check` | automated |
| JavaScript production dependency advisories | `npm run security:audit` in Docs Site Checks | automated |
| full product regressions | `python -m tests.run` | automated |
| clean first-run installation | required when onboarding or dependency claims change | conditional proof |
| desktop/mobile and light/dark rendering | manual rendered-site inspection | must become reproducible evidence |
| local search, navigation, edit links, and copy controls | manual browser inspection | must become reproducible evidence |
| production redirects, headers, sitemap, and robots | manual HTTP inspection | robots defect open |
| external URL health | none | adoption blocker |
| accessibility audit | none beyond semantic build output and manual inspection | adoption blocker |
| visual regression | none beyond manual screenshots | adoption blocker |
| performance budget | none | adoption blocker |

The portable system MUST automate the four adoption blockers before it is declared reusable. Other repositories MUST NOT weaken those gates while claiming the same standard.

## Required commands

From the repository root:

```bash
.venv/bin/python -m pytest -q tests/test_docs_surface.py
.venv/bin/python -m tests.run
```

From `docs-site`:

```bash
npm ci
npm run security:audit
npm run check
```

`npm run check` runs Markdown lint, assembles the corpus, and builds Docusaurus with broken-link failures enabled. The Python docs test owns source-linked semantic inventories, versions, local links and anchors, fence parsing, and abstract-example guards.

For a changed first-run path, also prove the documented install in a fresh environment. For layout, navigation, search, theme, or shared-system changes, execute the full parity protocol.

## Parity protocol

Parity preserves intended behavior, composition, and appearance. It does not preserve documented defects, security weaknesses, broken routes, or inaccessible behavior.

### Freeze the baseline

Record:

- baseline commit SHA and production URL
- lockfiles and tool versions
- maintained source inventory and source-to-route map
- sitemap URL set and HTTP status
- navbar, sidebar, footer, edit-link, search, and theme behavior
- CSS token and computed-layout values
- security headers, redirects, canonical metadata, and error responses
- representative screenshots
- every known defect and whether the candidate fixes, defers, or excludes it

Build the baseline before editing and retain it through acceptance:

```bash
cd docs-site
npm ci
npm run check
BASELINE_DIR="$(mktemp -d)"
cp -R build "$BASELINE_DIR/build"
```

### Build the candidate

Build with the same runtime, lockfile, environment variables, origin, and base path:

```bash
cd docs-site
npm run check
```

### Compare exact invariants

The following require equality unless the change has an approved expected-difference entry:

- maintained source set
- source-to-route mapping and public route set
- navbar, category order, sidebar behavior, footer, pagination, and edit targets
- search availability and representative result destinations
- metadata, canonical URLs, sitemap membership, headers, and status codes
- shared tokens, component geometry, responsive breakpoint, and theme behavior
- all non-target page content and screenshots

Compare HTML route filenames separately from content-hashed assets:

```bash
diff -u \
  <(cd "$BASELINE_DIR/build" && find . -type f -name '*.html' | sort) \
  <(cd build && find . -type f -name '*.html' | sort)
```

### Compare representative pages

The minimum matrix covers product home, one generated section index, one guide with code and a table, one reference page, one developer page, and one package README at:

- desktop `1440x900`, light
- desktop `1440x900`, dark
- mobile `390x844`, light
- mobile `390x844`, dark

For each state, compare the screenshot and rendered DOM. Confirm H1, intro, content width, typography, token colors, navbar height, sidebar state, table overflow, focus behavior, search, edit link, and absence of horizontal page overflow.

### Declare expected differences

Every intentional delta is listed in the slice or PR before acceptance with:

- affected route, viewport, and theme
- old and new behavior
- source of authority for the change
- screenshot or command evidence
- rollback effect

Unlisted differences fail parity. For this contract rewrite, only `/developer/documentation-system` content, its generated card summary on `/developer`, and the chunks or search index derived from that content may differ.

### Accept and retain rollback

Acceptance requires:

- all required commands green on the exact candidate
- zero unlisted route, interaction, metadata, console, or visual differences
- known defects fixed or linked to an explicit later slice; none silently accepted as parity
- candidate preview reviewed before production cutover
- previous production artifact retained until post-cutover smoke checks pass

## Rollout protocol

1. **Prove Limen.** Close the current font/CSP, robots, external-link, accessibility, visual-regression, and performance-gate gaps. Re-run the full parity matrix against the frozen Limen baseline.
2. **Extract the shared system.** Move shared theme, generator configuration, assembly algorithm, tests, and CI into one versioned package or preset. Move every product literal into a validated profile.
3. **Validate portability.** Run fixture builds for at least two deliberately different product profiles. The shared package must contain no Limen literals and must fail invalid or incomplete profiles.
4. **Pilot one unlike repository.** Choose a production repository with different source layout or package boundaries. Build behind a preview URL, map its content to the five-section architecture, and compare its old and candidate sites.
5. **Cut over the pilot.** Require its semantic, route, interaction, accessibility, security, performance, and visual gates. Retain the old artifact for rollback, then remove the old build path after the agreed expiry.
6. **Roll out sequentially.** Adopt the same pinned shared version in remaining repositories. Each repository supplies only its product profile, content map, source-backed semantic tests, and approved brand overrides.
7. **Operate centrally.** Release shared-system changes with compatibility notes and fixture proof. Upgrade repositories through dependency updates, never copied patches.

No repository enters rollout until Limen passes its blockers and the extracted shared layer contains no product-specific constants. No parallel permanent documentation implementation is allowed.

## Versioning and drift control

The shared system version and each product's content version are independent.

- Patch: compatible bug fix, gate repair, or token correction without route or profile-schema change.
- Minor: backward-compatible component, page contract, gate, or optional profile capability.
- Major: incompatible profile schema, information architecture, route policy, token contract, or required gate.

Every shared release includes:

- exact dependency and runtime support
- profile-schema compatibility
- changed page, visual, interaction, and gate contracts
- migration and rollback steps
- fixture and parity evidence

Product repositories pin an exact shared version and use automated dependency updates. A local override requires an owner, reason, linked issue, expiry, and proof that required accessibility, security, route, and quality behavior remains intact. Expired overrides fail CI.

## Known gaps before extraction

Limen is not yet ready to be published as the shared system because:

- the intended IBM Plex external font import conflicts with the production CSP
- `/limen/robots.txt` returns `404`
- external links are not mechanically checked
- accessibility is not audited by an automated required gate
- screenshots are not captured and compared by a reproducible visual-regression gate
- no performance budget blocks regression
- product-specific values remain embedded outside `product-docs.json`

These are implementation slices with their own task types and proof. They MUST be fixed in Limen and revalidated side by side before extraction.

## Review checklist

- Does every changed claim name or imply a current authoritative source?
- Does each page keep one role and the required composition for that role?
- Are product-specific claims confined to the product profile and content?
- Are prerequisites, outputs, failure boundaries, and next routes explicit?
- Do equivalent terms, units, examples, routes, and risk boundaries agree?
- Is every maintained source assembled exactly once?
- Do edit links target real source files?
- Are local and external links healthy?
- Do desktop/mobile and light/dark states match the shared visual contract?
- Are keyboard navigation, focus, contrast, landmarks, tables, and code usable?
- Are canonical metadata, sitemap, robots, redirects, headers, and errors correct?
- Do semantic tests, Markdown lint, security audit, and site build pass?
- Is every expected parity delta declared and every unexpected delta resolved?
- Is rollback available until post-cutover smoke checks pass?

## Read next

- [Docs hub](../README.md)
- [Developer home](README.md)
- [Writing docstrings](Writing-Docstrings.md)
- [Packaging](Packaging.md)
- [Security assurance case](Security-Assurance-Case.md)
