#!/bin/sh
set -eu
# Optional model setup and research; run from the recorded bundle root.
model=tabpfn-v2-classifier-v2_default.ckpt
if ! test -f "$model"; then
  curl --fail --location 'https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/f851f2a3c941544733b712d8c0f96dfae9b28862/tabpfn-v2-classifier-v2_default.ckpt' -o "$model"
fi
actual=$(openssl dgst -sha256 "$model" | awk '{print $NF}')
test "$actual" = cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49
limen validate compiled-research/tabpfn.yaml
limen run --no-progress-bar compiled-research/tabpfn.yaml
