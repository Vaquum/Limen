# AGENTS.md

## Task Type
- Every work unit must have exactly one primary `task_type`.
- Mixed requests must be decomposed before implementation.
- Unknown or ambiguous `task_type` means stop and report to Operator.
- `runtime_touching` is a separate discriminator. It adds proof obligations but does not change the primary `task_type`.
- Governance, routing, and authority changes are out of scope for WA and must be escalated.

| `task_type` | Intent | Forbidden change class |
| --- | --- | --- |
| `proof` | evidence and live validation only | runtime behavior change, deploy or env contract change, source or canonical semantics change |
| `planning` | publish approved slice or task scope without runtime change | runtime feature change, deploy or bootstrap change, governance contract change |
| `issue_authoring` | publish operator-specified issue without codebase change | repo file change, runtime change, governance contract change |
| `operator_surface` | authoritative operator or user visible surface | runtime dependency or bootstrap change, parser contract change, throughput-only tuning |
| `runtime_env` | execution environment, delivery, CI, and toolchain | operator or user surface change, source or canonical correctness change, throughput-only tuning without runtime contract change |
| `correctness` | make existing behavior true without new capability | new capability, operator-surface-only change, performance-only change |
| `performance` | improve cost, latency, or parallelism without semantic change | correctness fix disguised as speed work, runtime or bootstrap change, new feature |
| `capability` | add new supported system ability or surface | repair-only work, governance-only change, proof-only status work |

## Routing
1. Treat this file as the single routing surface for WA engineering work.
2. Classify the request to exactly one allowed `task_type` before any implementation.
3. Load universal control surfaces first: governance authority, task type catalog, contract applicability, reference or parser policy, branch and workflow discipline, task start gate, task end gate, compiler or rules-engine surfaces.
4. Then load task-specific surfaces plus any activated domain contracts for the touched subsystem.
5. Every touched authoritative surface must have routed contract coverage. No coverage means no work.
6. No catch-all route exists. No route means stop and report to Operator.
7. Review follow-up work inherits the governing `task_type` or stops.
8. `proof` must not be used as cover for implementation work.

## Governance
- WA may write the work plane only.
- WA may not touch governance-plane files.
- Mixed-plane diffs fail.
- Work must happen on a `wa/` branch, never on `main`.
- A local work branch must track the same-named remote branch.
- Local-only branches, extra worktrees, and detached-head write work are forbidden.
- At most one active local WA branch and one unmerged remote WA branch may exist at a time but can inherit branches from operator as long as one active local branch rule stands
- Dry compile is required before push.
- PR and authoritative CI compile are required before merge.
- Report-back or stand-down requires committed state, a clean tree, and the current branch head present on remote.
- Machine contracts override prose and repo folklore.

## Checks
**Before Work**
- No parallel mechanism without a migration end state and deadline.
- The governed path must not be harder than an unguarded shortcut.
- No recurring manual intervention in steady state.
- Enforcement must be mechanical, not merely documented.
- No unrequired knobs or abstractions.
- Leave one live path or an explicit migration path.
- Runtime-touching work must have declared proof coverage before implementation.

**During Work**
- Keep implementation scope inside the routed `task_type`.
- If the task mutates the repo: stage intended changes, run the staged or pre-push compiler gate, fix until clean, then commit.
- Runtime-touching work must produce canonical evidence, not narrative-only claims.
- Operator or user visible changes must remain mechanically enforced on authoritative surfaces.

**Before Completion**
- Run the required dry compile on the exact push candidate.
- Treat dry run as blocking but non-authoritative.
- Treat CI compile on the exact PR candidate tree as authoritative for merge.
- Do not claim done, ready, complete, or mergeable until the end gate passes.
- Claimed terminal state must be real, not merely prepared.
- Workflow discipline must be satisfied.
- Declared proof obligations must actually be met.

## MVP Gov Contracts
Minimal common contract shape preserved from the existing per-task developer contracts: `version`, `task_type`, `kind`, `purpose`, `hard_rules`.

```json
{
  "contracts": [
    {
      "version": 1,
      "task_type": "proof",
      "kind": "developer_task",
      "purpose": "evidence_and_live_validation_only",
      "hard_rules": [
        "evidence_and_live_validation_only",
        "no_contract_or_runtime_mutation",
        "report_reproducible_command_evidence"
      ]
    },
    {
      "version": 1,
      "task_type": "planning",
      "kind": "developer_task",
      "purpose": "publish_operator_approved_slice_scope_without_runtime_change",
      "hard_rules": [
        "approved_scope_only",
        "no_runtime_mutation",
        "output_as_execution_backlog"
      ]
    },
    {
      "version": 1,
      "task_type": "issue_authoring",
      "kind": "developer_task",
      "purpose": "publish_operator_specified_issue_without_codebase_change",
      "hard_rules": [
        "operator_spec_scope_only",
        "no_codebase_mutation",
        "include_repro_steps_and_acceptance"
      ]
    },
    {
      "version": 1,
      "task_type": "operator_surface",
      "kind": "developer_task",
      "purpose": "authoritative_operator_or_user_visible_surface",
      "hard_rules": [
        "surface_contract_consistency_required",
        "user_visible_text_requires_contract_alignment",
        "no_hidden_runtime_assumptions"
      ]
    },
    {
      "version": 1,
      "task_type": "runtime_env",
      "kind": "developer_task",
      "purpose": "execution_environment_delivery_ci_and_toolchain_contract",
      "hard_rules": [
        "toolchain_contracts_only",
        "no_business_logic_change",
        "deterministic_environment_repro_required"
      ]
    },
    {
      "version": 1,
      "task_type": "correctness",
      "kind": "developer_task",
      "purpose": "make_existing_behavior_true_without_new_capability",
      "hard_rules": [
        "behavior_preservation_except_fix_scope",
        "regression_protection_required",
        "no_new_capability_surface"
      ]
    },
    {
      "version": 1,
      "task_type": "performance",
      "kind": "developer_task",
      "purpose": "improve_cost_latency_or_parallelism_without_semantic_change",
      "hard_rules": [
        "semantic_equivalence_required",
        "metric_delta_evidence_required",
        "no_behavioral_drift"
      ]
    },
    {
      "version": 1,
      "task_type": "capability",
      "kind": "developer_task",
      "purpose": "new_supported_system_ability_or_surface",
      "hard_rules": [
        "contract_and_rule_registration_required",
        "acceptance_gate_updates_required",
        "end_to_end_validation_required"
      ]
    }
  ]
}
