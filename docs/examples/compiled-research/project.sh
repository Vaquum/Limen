#!/bin/sh
set -eu
# Run from the bundle root. This campaign has its own Git history and local backup.
# region lifecycle
bundle=$(pwd)
limen --version
limen list-templates
mkdir -p results/compiled-project
cp compiled-research/limen.toml results/compiled-project/limen.toml
cp compiled-research/project.yaml results/compiled-project/research.yaml
cp -R "$bundle/inputs" results/compiled-project/inputs
printf 'inputs/\nresults/dev/\n' > results/compiled-project/.gitignore
git init --quiet --bare results/compiled-backup.git
git init --quiet results/compiled-project
cd results/compiled-project
git config user.name "Limen documentation"
git config user.email "docs@example.invalid"
git config commit.gpgsign false
limen init scaffold.yaml --template logreg_binary
limen validate research.yaml
limen profile research.yaml
limen run --dry-run research.yaml
limen commit research.yaml --message "docs: record the research definition"
limen ls
for definition in manifests/committed/*.yaml; do
  identity=$(basename "$definition" .yaml)
done
limen lineage "$identity"
limen fork "$identity" recorded-child
cp "$bundle/compiled-research/project_child.yaml" manifests/recorded-child-production.yaml
limen commit manifests/recorded-child-production.yaml --parent "sha256:$identity" --message "docs: record the child definition"
limen reindex
limen run --no-progress-bar "manifest://sha256:$identity"
limen backup
cd "$bundle"
limen new results/compiled-restored --from "$bundle/results/compiled-backup.git"
# endregion lifecycle
