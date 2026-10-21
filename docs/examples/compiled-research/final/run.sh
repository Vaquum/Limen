#!/bin/sh
set -eu
# region final_assessment
git init --quiet compiled-research/final
reservation=$(limen reserve compiled-research/final/holdout.yaml)
limen run --no-progress-bar compiled-research/final/holdout.yaml
limen assess compiled-research/final/holdout.yaml --results results/dev/compiled-holdout --reservation "$reservation"
# endregion final_assessment
