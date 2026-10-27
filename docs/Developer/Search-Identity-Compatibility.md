# Search identity and recovery compatibility design

This is the design decision for [#1281](https://github.com/Vaquum/Limen/issues/1281).
It specifies a future versioned identity protocol. It does not change the shipped
hashes, parameter admission, checkpoint codec or supported recovery paths.

## Prerequisites

Read the [search reference](../Search-Reference.md) and retain a matching released
wheel and recorded run before evaluating any recovery-format change.

## Existing behavior

Ordinary search uses `limen._parameter_identity.execution_hash`: SHA-256 of sorted
JSON after removing underscore-prefixed metadata keys, with `default=str`.
Choice admission rejects collisions under that identity. Domain mutations use
that same identity. This preserves existing IDs but does not make arbitrary
Python values portable or lossless.

Factorized search separately uses the typed `parameter_key` encoding and its
saved factorization metadata. Grid shuffle versions describe traversal, rather
than the meaning of a parameter value. These are distinct existing protocols;
neither may be silently renamed to the proposed protocol.

Checkpoint JSON, recorded round parameters and CSV are also separate surfaces.
Changing a hash without changing parameter serialization can turn a declared
tuple into a list, or make completed candidates appear unseen. Current recovery
refusals and typed reconstruction must remain in force until a complete
replacement has passed the compatibility proof below.

## Canonical value model

Lossless means injective over the following declared value model, with the same
encoding across processes and supported interpreter versions. It does not mean
serializing every possible Python object.

| Value | Canonical representation and recovery |
|---|---|
| `None`, `bool`, `int`, `str` | Separate explicit type tags; integers use exact decimal strings |
| Finite Python `float` | Exact hexadecimal representation, preserving signed zero |
| `list`, `tuple` | Separate tags and recursively encoded ordered members |
| Mapping with string keys | A mapping tag and key-sorted pairs; insertion order is not identity |
| YAML scalar wrappers | Normalize to the YAML scalar's built-in semantic type; formatting and anchors do not change identity |
| NumPy boolean, integer and floating scalars | Explicit dtype plus exact scalar value; preserve the dtype on decode |
| NaN, infinity, arrays, sets, non-string mapping keys, `Decimal`, enums and arbitrary objects | Refuse with the parameter path before publication |
| Callable choices in persisted runs | Refuse in the initial protocol; an import name alone does not identify executable code |

In-process Python work may retain invocation-local callable/object choices under
its existing contract. It must not claim that those identities can be resumed
in another process. Extending the persisted model requires a separately reviewed
protocol version, not `str`, `repr`, pickle or a best-effort fallback.

The codec uses unambiguous tagged arrays, not user dictionaries with reserved
keys. Canonical UTF-8 JSON has sorted mapping pairs, explicit type tags and no
NaN tokens. A value's identity is derived from its canonical codec output.
Encode/decode/encode must preserve the bytes; decode must reconstruct the
declared Python value type. All domain comparison, mutation and cache keys must
use the same active value model.

## Version and ownership

Use an explicit `param_identity_version` in run metadata and checkpoint strategy
state, plus an explicit parameter-codec version. Run and checkpoint markers must
agree before any state is adopted or artifact is written. Unknown versions fail
closed. Missing markers identify the legacy protocol, not the new default.

Inventory the factorized marker already shipped before assigning version names.
An existing factorized directory stays on its existing typed protocol. Ordinary
legacy directories stay on their historical JSON protocol. A new protocol gets
a distinct name; it is not inferred from the installed package version.

Keep public round IDs as 64 lowercase hexadecimal SHA-256 characters. Domain
separate the new preimage with a fixed protocol tag and canonical encoding.
Underscore-prefixed execution metadata remains excluded. Round identity,
domain traversal digest and filter sampling identity each need explicit domain
separation and version-aware callers. Hash domain separation does not claim
mathematical collision impossibility for SHA-256.

An experiment's protocol is fixed for its lifetime. There is no in-place
re-keying, mixed-protocol append, CSV-based migration or rewriting of historical
round IDs. A new run may use the new protocol after implementation is qualified.
An old run continues only under its original supported protocol and codec; an
unrecoverable historical value is refused, not guessed.

Persisted custom `compute_param_hash` overrides require an explicit stable
protocol identifier and a matching codec/recovery contract. An undeclared
override cannot create a new-format persisted run. Existing invocation-local
extension behavior stays separate.

## Persistence and consumers

One delivery must update and test all of these consumers together:

| Surface | Required invariant |
|---|---|
| Parameter admission, `ParamDomain`, filters and injections | Equality, collision rejection, removal and kept values agree with the active identity |
| Grid, Random and Sobol strategy state | Seen sets, domain digests, traversal version and cursor retain their original meaning |
| MSQ priority queue and checkpoint domain | Decode the original types before validation or iteration |
| `round_params` and CSV | The typed record is authoritative; CSV remains presentation and must never be parsed to invent a typed value |
| Serial, parallel and filter recovery | Exactly the committed occurrences continue; completed work is neither repeated nor omitted |
| Factorized signal/preparation keys | Both cache layers agree with the active codec and preserve their existing independent keys |
| Trainer, Cohort and saved selectors | Existing `(manifest_id, permutation_id)` handles continue to resolve |
| Walk-forward evidence | Fold records keep the same round identity through reconstruction |
| Guarded holdout SQLite and checkpoints | Project-level `round_id` references stay valid without rewriting research history |

Manifest content addresses are outside this change. Their canonicalizer and
stored IDs retain their existing protocol. Adding search identity versioning
must not modify manifest lineage or reinterpret a manifest hash.

## Upgrade and downgrade

New-format directories must be rejected by the previous supported release
before it writes anything. Adding an ignored JSON field is insufficient.
Qualify a new directory/round-storage format that the previous reader already
refuses; explicitly verify serial, parallel and CLI intake ordering. If any old
entry point writes before refusing, the new writer cannot ship for that path.
Do not claim that a new marker alone establishes downgrade safety.

The new reader must distinguish ordinary legacy, existing factorized and new
directories. Unknown or inconsistent identity, codec, storage and traversal
markers refuse with a diagnostic naming the incompatible surface. Loading an
archive must not implicitly upgrade its enclosed run.

## Implementation admission and proof

Implement the protocol only through a separate PRD and issue after recording
the actual previous release and its artifact hashes. That implementation must
provide the following evidence, rather than treating this design as proof:

1. Record seeded Grid (both traversal versions), Random and Sobol ID sequences
   and domain digests for shipped applicable manifests/native SFDs. Preserve
   existing-protocol goldens byte for byte across the supported Python matrix.
2. Create actual interrupted directories with the previous release using the
   repository's recorded observations. Resume under the new release through
   serial, parallel, factorized, walk-forward and guarded-holdout routes. Verify
   original IDs, typed model inputs, no repeated completed work and equivalent
   results; verify Trainer and Cohort handles separately where supported.
3. Run the previous release against new-format directories through each
   supported entry point. Assert refusal and unchanged names, bytes and mtimes.
4. Test nested type distinctions, NumPy dtypes, signed zero, YAML semantic
   scalars and all explicit refusals. Compare canonical bytes and IDs in
   subprocesses with different `PYTHONHASHSEED` values.
5. Exercise domain mutation, injected priority values, cache hits, round-record
   decode and failed-round recovery under each supported protocol. An unknown
   marker or mismatched codec must fail before fit or artifact mutation.
6. Pass all existing required checks without raising typing, fallback, module
   or runtime budgets. Measure the added proof cost before requesting merge.

The smallest safe delivery is the codec and all its identity consumers together.
A serializer-only or hash-only patch cannot satisfy this design. Experiment
archives can preserve existing bytes and historical protocols independently;
they do not need to introduce the new identity format.
