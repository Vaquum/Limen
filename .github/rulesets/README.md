# Ruleset snapshot

`main.json` mirrors the live `Protect-Main` ruleset. The `pr_checks_ruleset`
gate fails any PR while the two differ.

`main` ALSO carries a legacy branch protection rule with the same
required-check list. It is not tracked in this repository. Any
required-check swap must update BOTH the live ruleset and the legacy
protection (via the rulesets API and
`branches/main/protection/required_status_checks`), or the PR can
satisfy one enforcement layer but never both (observed on #1138).
