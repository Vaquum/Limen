#!/bin/sh
set -eu
# A data-dependent sanity refusal is an observed contract result, never a substituted model.
mkdir -p results/target-contracts
for definition in compiled-research/target_forwardvolnormalizedreturn.yaml compiled-research/target_volnormalizedreturn.yaml; do
  name=${definition##*/}
  status=0
  limen run --no-progress-bar "$definition" > "results/target-contracts/$name.log" 2>&1 || status=$?
  cat "results/target-contracts/$name.log"
  if test "$status" -ne 0; then
    grep -F 'Parkinson-to-close volatility ratio' "results/target-contracts/$name.log"
  fi
done
