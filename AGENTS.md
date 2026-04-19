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
{
  "github_rules": {
    "issue_authoring": {
      "issue_write_requires_explicit_operator_request": true,
      "issue_write_requires_operator_defined_scope_or_source_material": true,
      "issue_content_must_not_add_agent_invented_scope_or_requirements": true,
      "repo_files_must_not_be_modified": true,
      "codebase_change_forbidden": true
    },
    "slice_issue_authority": {
      "planning_converts_operator_approved_work_into_authoritative_github_slice_issue_before_implementation": true,
      "chat_level_plans_are_not_official_until_published_into_authoritative_github_slice_issue": true,
      "authoritative_new_or_replanned_slice_surface": "github_issue",
      "authoritative_slice_issue_template_path": ".github/ISSUE_TEMPLATE/slice.yml",
      "slice_number_must_follow_authoritative_issue_sequence": true,
      "ordered_github_issue_programs_are_authoritative_for_planning_when_present": true,
      "only_next_ordered_item_may_expand_into_new_slice_unless_explicit_replanning": true,
      "planning_issue_must_not_restate_generic_task_type_or_global_governance_rules": true,
      "future_runtime_touching_slice_requires_explicit_observability_scope": true,
      "future_runtime_touching_slice_requires_canonical_proof_coverage_section": true,
      "proof_coverage_issue_section_label": "Proof Coverage",
      "authoritative_issue_or_checkbox_tracker_must_be_updated_when_it_governs_slice_order": true,
      "slice_must_define_capability_proof_and_guardrail_scope_before_implementation": true,
      "slice_is_not_ready_until_authoritative_issue_is_in_sync": true,
      "completed_items_must_be_checked_off_in_issue": true,
      "closeout_evidence_must_be_linked_not_duplicated": true,
      "new_governance_or_closeout_work_must_be_recorded_in_authoritative_issue_before_implementation": true,
      "issue_authoritative_slice_repo_planning_residue_forbidden": true,
      "required_issue_fields_for_new_or_replanned_slice": [
        "Slice ID",
        "Slice Title",
        "Primary Task Type",
        "Routing and Dependencies",
        "Objective, Baseline, and Scope",
        "Design",
        "Runtime and Observability Scope",
        "Proof Coverage",
        "Capability",
        "Proof",
        "Guardrails",
        "Step Register",
        "Risks and Open Questions",
        "Done Means",
        "Author Checks"
      ],
      "required_done_means_checkboxes": [
        "Capability complete",
        "Proof complete",
        "Guardrails complete",
        "Required gates green",
        "Required docs updated",
        "Required closeout state and evidence links are recorded in this issue"
      ]
    },
    "pull_request_discipline": {
      "pull_request_first_local_test_suite_next": true,
      "protected_branch": "main",
      "branch_prefix": "wa/",
      "repo_mutation_on_main_branch_forbidden": true,
      "repo_mutation_in_detached_head_forbidden": true,
      "repo_mutation_in_temp_or_ad_hoc_worktree_forbidden": true,
      "repo_mutation_outside_single_allowed_local_work_branch_forbidden": true,
      "only_one_unmerged_local_work_branch_allowed_at_a_time": true,
      "only_one_agent_authored_open_pr_allowed_at_a_time": true,
      "local_work_branch_must_track_remote_branch": true,
      "local_work_branch_must_be_up_to_date_with_remote_branch_before_repo_mutation": true,
      "local_work_branch_must_be_up_to_date_with_remote_branch_before_terminal_claim": true,
      "repo_mutation_requires_commit_and_push_before_terminal_claim": true,
      "repo_mutating_task_requires_open_pr_before_terminal_claim": true,
      "terminal_report_must_end_with_commit_hash_and_open_pr_url": true,
      "commit_message_must_follow_conventional_commits": true,
      "pr_title_must_follow_conventional_commits": true,
      "pr_title_must_not_mention_codex": true,
      "draft_pr_for_review_forbidden": true,
      "max_local_branches_per_role": 1,
      "max_unmerged_remote_branches_per_role": 1,
      "max_open_prs_per_role": 1,
      "direct_push_to_protected_branch": false
    },
    "review_flow": {
      "required_review_request": true,
      "must_wait_for_review_outcome": true,
      "runtime_touching_slice_pr_requires_reviewable_issue_proof_coverage": true,
      "runtime_touching_slice_pr_requires_linked_terminal_proof_plan": true,
      "must_resolve_all_conversations_before_rerequest": true,
      "must_rerequest_review_after_conversation_resolution": true,
      "required_approvals": 1
    }
  }
}
