#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-}"
case "${DATASET}" in
  gabor)
    python -m jacaccess.io.download openneuro \
      --dataset ds005273 \
      --destination data/raw/gabor
    python -m jacaccess.io.download finalize \
      --raw-root data/raw/gabor \
      --output data/manifests/gabor_raw.json
    ;;
  somato)
    python -m jacaccess.io.download osf \
      --project hqkym \
      --destination data/raw/somato
    python -m jacaccess.io.download finalize \
      --raw-root data/raw/somato \
      --output data/manifests/somato_raw.json
    ;;
  kronemer)
    : "${NITRC_COOKIE:?Set NITRC_COOKIE after authenticating with NITRC}"
    python -m jacaccess.io.download manifest \
      --manifest configs/data_sources/kronemer_downloads.tsv \
      --destination data/raw/kronemer \
      --cookie "${NITRC_COOKIE}"
    python -m jacaccess.io.download finalize \
      --raw-root data/raw/kronemer \
      --output data/manifests/kronemer_raw.json
    ;;
  *)
    echo "Usage: $0 {gabor|somato|kronemer}" >&2
    exit 2
    ;;
esac

