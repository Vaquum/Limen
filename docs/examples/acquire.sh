#!/bin/sh
set -eu
# Acquire one publication atomically by identity; never combine a new wheel with old examples.
# region acquire
destination=${1:?Usage: sh acquire.sh NEW_DIRECTORY}
origin=${LIMEN_DOCS_ORIGIN:-https://docs.vaquum.fi/limen}
mkdir "$destination"
cd "$destination"
curl --fail --silent --show-error --location "$origin/publication.json" -o publication.json
release=$(jq -er '.release' publication.json)
for artifact in examples wheel; do
  path=$(jq -er --arg artifact "$artifact" '.artifacts[$artifact].path' publication.json)
  expected=$(jq -er --arg artifact "$artifact" '.artifacts[$artifact].sha256' publication.json)
  case "$path" in evidence/*) ;; *) echo 'Publication artifact is outside evidence/' >&2; exit 1 ;; esac
  case "$path" in *..*|*://*|*\?*|*\#*) echo 'Invalid publication artifact path' >&2; exit 1 ;; esac
  file=${path##*/}
  curl --fail --silent --show-error --location "$origin/$path" -o "$file"
  actual=$(openssl dgst -sha256 "$file" | awk '{print $NF}')
  test "$actual" = "$expected" || { echo "Changed $artifact artifact" >&2; exit 1; }
  if test "$artifact" = examples; then bundle=$file; else wheel=$file; fi
done
tar -xzf "$bundle"
python3 -m venv .venv
.venv/bin/python -m pip install "./$wheel[data,stats,boosting]"
printf 'Acquired %s. Run from this directory after: . .venv/bin/activate\n' "$release"
# endregion acquire
