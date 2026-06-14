# Documentation System Contract

## Purpose

This page defines how Limen documentation should be structured, written, built, and improved. It is the operating contract for the docs overhaul and the reference point for every later documentation change.

The goal is not only page-level correctness. The goal is one coherent Limen documentation product.

## What 10/10 Means

For Limen, 10/10 docs means the full system is:

- accurate to the code and current Vaquum architecture
- direct entry path for a new user
- deep enough for serious research and contributor work
- coherent across product pages, reference pages, and package READMEs
- grounded in real runnable workflows and real artefacts
- ready to power a standalone Limen docs site now and a Vaquum docs portal later

## Product Docs Model

### Current Site Direction

Limen should get its own standalone docs site from this repository.

- The initial site build target is Docusaurus.
- The site should be built in this repository.
- The site should deliver a complete Limen-only docs experience.
- The site should work as a standalone product site first.
- The site should later be portable into a broader Vaquum docs system without rewriting the content model.
- Canonical content should remain owned by repository markdown, not by site-only copies of the docs.

### Future Vaquum Docs Direction

The long-term target is `docs.vaquum.fi` as the Vaquum documentation entry point.

- `docs.vaquum.fi` should present all Vaquum product docs: Origo, Limen, Nexus, Praxis, and Veritas.
- Each product should still feel discrete, self-contained, and product-native.
- The first Vaquum-wide version should behave like a portal to product docs, not like one merged blob of markdown.
- The Limen docs system should therefore be designed to plug into a future Vaquum docs shell without losing its own identity.

## Canonical Source Rules

Limen documentation should have one ownership model.

- [README.md](../../README.md) is the product home page and first-success entry point.
- [docs/README.md](../README.md) is the canonical public docs hub.
- `/docs` is the canonical source for public concepts, workflows, and API/reference pages.
- `/docs/Developer` is the canonical source for contributor and maintainer process docs.
- package `README`s under `/limen` are orientation pages for module ownership and boundaries, not the main public reference.
- examples should be derived from real runnable flows in this repository, not imaginary or hand-waved pseudo-usage.

Content should be authored once whenever possible. If the same explanation appears in multiple places, one page should be canonical and the others should route to it.

## Information Architecture

The docs site should be organized into five top-level sections:

- `Overview`
- `Guides`
- `Reference`
- `Developer`
- `Packages`

The source files can remain in their current repository layout, but the built site should present them through this information architecture.

### Section Responsibilities

- `Overview` explains what Limen is, what it is not, and how the whole system fits together.
- `Guides` teach workflows and tasks from start to finish.
- `Reference` documents surfaces, conventions, arguments, outputs, and edge cases.
- `Developer` documents contribution, release, maintenance, and internal documentation rules.
- `Packages` explains module ownership, boundaries, entry points, and where to read next.

## Narrative Spine

Every major public page should reinforce the same core Limen story:

1. Limen turns Bitcoin market data into searchable signals, backtested outcomes, and decoder cohorts.
2. Data enters through native Limen data access or external OHLC-style data.
3. An experiment is defined through an SFD, manifest-driven by default.
4. Universal Experiment Loop executes a parameter search across the chosen search space.
5. Log, benchmark-style analytics, and backtest outputs show what happened and why.
6. Trainer and Cohort turn finished experiment outputs into reusable downstream artefacts.
7. Trade decisioning and execution happen outside Limen, downstream in other Vaquum systems.

If a page does not help a reader understand its place in that story, it should route clearly to the pages that do.

## Register And Writing Rules

All Limen documentation should use the same register:

- precise
- technical
- concise
- accessible to an informed new user
- direct rather than academic
- product-truthful rather than hype-driven

### Writing Rules

- Start with what the thing is and why a reader would use it.
- Prefer concrete behavior over abstract framing.
- Keep theory only where it directly improves practical understanding.
- Explain current surface area honestly; do not imply future behavior as present behavior.
- Prefer examples that show inputs, outputs, and artefacts.
- Do not use unexplained internal jargon.
- Do not duplicate large sections of content across pages.
- End pages with explicit reading routes or next steps when they affect task completion.

## Page Types And Required Blocks

Every page should fit one primary page type.

### Home Page

Purpose: product framing and first success.

Required blocks:

- what Limen is
- what Limen is not
- capability summary
- first successful workflow
- explicit routes into the rest of the docs

### Docs Hub

Purpose: route readers by task and audience.

Required blocks:

- system overview
- reading order by user type
- high-level architecture map
- explicit routes into guides, reference, developer docs, and package docs

### Guide

Purpose: teach a job or workflow from start to finish.

Required blocks:

- what this guide covers
- prerequisites
- current scope
- at least one concrete example
- expected artefacts or outputs
- related pages or next steps

### Reference

Purpose: document an interface or surface comprehensively and predictably.

Required blocks:

- short intro and scope
- conventions or naming rules
- structured entry documentation
- output columns or return behavior where relevant
- edge cases or caveats where relevant

### Developer Page

Purpose: guide contributors and maintainers.

Required blocks:

- page purpose
- required reading or prerequisites
- process or checklist
- failure cases or review notes where relevant
- linked related maintenance pages

### Package README

Purpose: orient readers inside a module without replacing canonical public docs.

Required blocks:

- what the package owns
- what it does not own
- key entry points
- major dependencies or adjacent modules
- link to canonical public docs

## Navigation And Cross-Link Rules

Navigation should reduce guesswork.

- The home page and docs hub must both provide reading paths by task.
- Large pages should be indexed near the top.
- Public workflow pages should link forward through the narrative spine.
- Package READMEs should link outward to canonical docs rather than trying to become their own mini-sites.
- Cross-links should prefer the next page a reader should open, not every vaguely related page.

## Terminology Rules

Use one terminology set across the whole docs system.

- Product name: `Limen`
- Template unit: `SFD` or `Single-File Decoder`
- Declarative SFD spec: `Manifest`
- Experiment engine: `Universal Experiment Loop` or `UEL`
- Cohort selection method: `selector`
- Reusable trained inference object: `Sensor`
- Data access class: `HistoricalData`

### Naming Rules

- Use `Loop` only when it is part of the actual component name `Universal Experiment Loop`.
- Do not describe Nexus decision logic as part of Limen.
- Do not describe Origo, Praxis, or Veritas responsibilities as if they live inside Limen.
- Keep `Bitcoin-only` language exact and consistent with the actual product boundary.

## Example And Artefact Rules

Examples should be operationally real.

- Prefer examples that can be run locally in this repository.
- Prefer examples validated against actual Limen artefacts.
- When an example depends on warnings, limitations, or approximations, say so explicitly.
- Show output tables, files, or artefacts where they are important to understanding the workflow.
- Avoid examples that accidentally teach readers to optimize only for a single metric.

## Site Build Rules

These rules should guide the site build implemented in the next slice.

- Limen should get a standalone docs site built in this repository.
- The initial implementation should use Docusaurus.
- The site should support local development and static build.
- The site should support both standalone deployment and later subpath deployment.
- The site base URL should be environment-driven so the same content can support both modes.
- Search should work across the full Limen docs corpus.
- Broken internal links should fail the build.
- The site navigation should reflect the five top-level sections in this contract.

## Future Vaquum Portal Contract

The Limen docs system should expose a minimal product-docs contract that can later be consumed by a Vaquum-wide docs portal.

That contract should include:

- product id
- product name
- short product tagline
- current docs version label
- deployment base path
- primary navigation sections
- source repository URL

This does not mean Limen should wait for a Vaquum-wide shell. Limen should be excellent as a standalone docs product first.

## Rewrite Slices

The overhaul should be tracked in the following order.

| Slice | Name | Scope | Definition of Done |
| --- | --- | --- | --- |
| 1 | Docs System Contract | Define structure, voice, page types, ownership, navigation model, site boundary, and rewrite slices. | This contract exists, is linked from developer docs, and is accepted as the operating manual for later slices. |
| 2 | Docs-Site Build | Add the site build, docs assembly model, product metadata, local dev/build/check commands, and navigation shell. | The Limen docs site builds locally, renders the current corpus, and enforces link integrity. |
| 3 | Top-level narrative | Rewrite the product home and docs hub so Limen has one entry story and reading flow. | A new user can enter the docs without guessing what to read next. |
| 4 | Core Workflow Guides | Rewrite data, bars, SFD, manifest, and UEL pages as one connected workflow layer. | A reader can go from data to running an experiment using only the guide layer. |
| 5 | Analysis And Outcomes | Rewrite log, benchmark, backtest, trainer, cohort, metrics, and CFR pages. | A reader can understand what Limen produces after a run and how to interpret it. |
| 6 | Reference Layer | Rewrite indicators, features, transforms, and scalers as coordinated reference pages. | The large reference pages are scannable, consistent, and trusted. |
| 7 | Developer layer | Rewrite contributor, release, and versioning docs for current practice. | A contributor can follow the maintenance workflow without external tribal knowledge. |
| 8 | Package README Alignment | Align package READMEs with the same contract and route them to canonical docs. | Package READMEs feel like part of one system rather than isolated notes. |
| 9 | Final cohesion pass | Sweep the entire corpus for terminology, duplication, examples, links, navigation, and consistency. | The docs read as one coherent product and meet the acceptance bar. |

## Acceptance bar for the overhaul

The overhaul should be considered complete when all of the following are true:

- a new user can reach a first meaningful experiment path without guessing
- a serious user can understand how Limen fits into the broader Vaquum architecture
- a contributor can find the canonical page for any subsystem quickly
- examples are grounded in real Limen runs and artefacts
- large reference pages have explicit navigation and code-true claims
- the standalone Limen docs site feels complete
- the same docs system can later plug into a Vaquum-wide docs portal without conceptual rework

## How To Use This Page

Before rewriting any major docs slice:

- confirm the target page type
- confirm where the page sits in the narrative spine
- confirm whether the page is canonical or secondary
- confirm which slice the change belongs to

If a proposed docs change conflicts with this contract, update this page first or explicitly document the exception.
