#!/bin/sh
set -eu
# region final_assessment
git init --quiet compiled-research/final-event
reservation=$(limen reserve compiled-research/final-event/holdout.yaml)
limen run --no-progress-bar compiled-research/final-event/holdout.yaml
limen assess compiled-research/final-event/holdout.yaml --results results/dev/compiled-holdout_event --reservation "$reservation"
# endregion final_assessment
