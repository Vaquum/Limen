# Edit and verify the research book

The documentation connects research decisions to executable projects, current source contracts, and recorded results. Prose editing preserves those connections; changes to examples or structure require source-level validation. The public reader and authenticated editor have separate publication lifecycles.

The [Python project](../examples/python-research/README.md) and [Compiled project](../examples/compiled-research/README.md) are the canonical worked examples. Their complete sources, inputs, and execution evidence supply the examples displayed throughout the book.

## Prerequisites

Source contributions require a current checkout, the repository's supported Python environment, and the locked documentation dependencies. Changes follow the issue, branch, pull request, and review requirements in [CLAUDE.md](https://github.com/Vaquum/Limen/blob/main/CLAUDE.md). The authenticated editor additionally requires current GitHub write permission and an activated deployment. [Documentation System](Documentation-System.md#prose-editing) records configuration and activation status.

## The reader supports sequential and reference work

Book navigation follows the research decisions in chapter order. The interface control shows Python Research, Compiled Research, or both. A direct passage or code link reveals its applicable interface. Responsive Contents navigation supplies the same chapter and heading destinations on smaller screens.

Reference links open their contract while preserving the chapter passage. Return to chapter restores reading position and focus. Search locates passages across Book, Reference, and Developer. Code controls preserve source bytes when copying and provide line/range links for a specific excerpt.

Reading position is local to the browser and scoped to product and content identity. A shared link should identify the intended passage or code range explicitly. The reader's source revision and execution evidence identify the document and package being read.

## The editor changes existing prose passages

The editor displays its branch and source revision before editing. Existing prose passages can be changed in place. Code examples, generated results, document structure, and protected source regions require repository edits. Branch previews identify unexecuted changes separately from historical execution evidence.

The authoring sequence has four steps:

1. Edit the required prose passages and inspect the accumulated changes in Review.
2. Use Undo for local draft changes or download the pending draft before leaving the browser.
3. Commit the reviewed changes to the editor's origin branch. Additional commits accumulate on that branch.
4. Open the issue-bound pull request to start checks and review. Completion follows the repository's normal merge requirements.

Local Undo does not revert a remote commit. Source movement, a missing branch, or a conflicting pull request requires reconciliation. Drafts belong to the signed-in writer and remain local to that browser. Sign-out and explicit discard follow the editor's displayed draft handling.

## Source changes preserve one canonical contract

A book chapter begins with the research decision and develops the relevant executable case. A reference page owns the complete API or configuration contract. Related chapters link to that contract while explaining its consequences for their own research decision.

Example fences bind exact named regions from the canonical project. Results bind fields from successful scenario outputs or immutable retained observations. Source edits must update the corresponding excerpts and assertions together. Hand-edited output cannot replace an actual model or scenario execution.

New chapters also require a route and chapter order in `docs-site/reader/content.json`, a maintained-page entry and scenario contexts in `docs/examples/catalogue.json`, and links in the reading sequence. Applicable interfaces need substantive examples and visible connections to their complete project. The [documentation contract](Documentation-System.md) defines binding syntax, source ownership, voice, and evidence requirements.

## Verification follows the changed behavior

Binding and source checks run before the complete project proof. The complete proof executes both canonical projects against the matching installed wheel outside the checkout. The reader build consumes that evidence from a clean source revision. Browser checks then verify navigation, interface selection, references, code controls, accessibility, and rendered routes.

The commands and required order are maintained in [Validate the reader and research](Documentation-System.md#validate-the-reader-and-research). A contributor records the actual command outcomes and addresses every required check and review thread. A content change that modifies a research claim also requires the assertion that establishes the new claim.

## Publication identifies the released package

Merging documentation completes the source change. Public publication additionally requires release-bound project evidence, public artifact identities, anonymous verification, and accepted activation. The editor deployment follows its separate maintainer process and does not activate a public book release.

[Release policy](Release-Policy.md) and [Making a release](Making-Release.md) own package delivery. [Documentation System](Documentation-System.md) owns reader publication, verification, and rollback. The public publication manifest establishes which release and source revision are active.
