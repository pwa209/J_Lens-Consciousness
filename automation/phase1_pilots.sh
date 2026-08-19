#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
cd "$ROOT"
source .venv/bin/activate

mkdir -p \
  data/downloads/pilots \
  data/raw/gabor \
  data/raw/somato \
  data/raw/kronemer \
  data/manifests \
  results/source-inspection/inventories

check_disk() {
  local free_kib
  free_kib=$(df --output=avail -k /root/autodl-tmp | tail -1 | tr -d ' ')
  if (( free_kib < 524288000 )); then
    echo "Refusing acquisition: less than 500 GiB scratch remains" >&2
    return 70
  fi
}

check_disk

if [[ ! -f data/downloads/pilots/.http_downloads_complete ]]; then
  python -m jacaccess.io.ranged_download \
    --manifest configs/data_sources/pilot_downloads.tsv \
    --destination data/downloads/pilots \
    --connections 8 \
    --minimum-free-gib 500
  touch data/downloads/pilots/.http_downloads_complete
fi

if [[ ! -f data/raw/gabor/sub-10/.acquisition_complete ]]; then
  mkdir -p data/raw/gabor/sub-10
  aws s3 sync --no-sign-request \
    s3://openneuro.org/ds005273/sub-10/ \
    data/raw/gabor/sub-10/
  for name in dataset_description.json CHANGES; do
    aws s3 cp --no-sign-request \
      "s3://openneuro.org/ds005273/$name" \
      "data/raw/gabor/$name"
  done
  touch data/raw/gabor/sub-10/.acquisition_complete
fi

if [[ ! -f data/raw/somato/.pilot_extraction_complete ]]; then
  unzip -Z1 data/downloads/pilots/somato/DATASET_PREPROCESSED.zip \
    > results/source-inspection/inventories/somato-preprocessed.txt
  unzip -q -n data/downloads/pilots/somato/DATASET_PREPROCESSED.zip \
    -d data/raw/somato
  touch data/raw/somato/.pilot_extraction_complete
fi

for archive in 223_RP_EEG.tar 238_NRP.tar; do
  participant=${archive%.tar}
  destination="data/raw/kronemer/$participant"
  sentinel="$destination/.pilot_extraction_complete"
  mkdir -p "$destination"
  if [[ ! -f "$sentinel" ]]; then
    tar -tf "data/downloads/pilots/kronemer/$archive" \
      > "results/source-inspection/inventories/${archive%.tar}.txt"
    python automation/selective_tar_extract.py \
      --archive "data/downloads/pilots/kronemer/$archive" \
      --destination "$destination" \
      --receipt "results/source-inspection/inventories/${archive%.tar}.json"
    touch "$sentinel"
  fi
  check_disk
done

python -m jacaccess.io.download finalize \
  --raw-root data/raw/gabor \
  --output data/manifests/gabor_pilot_raw.json
python -m jacaccess.io.download finalize \
  --raw-root data/raw/somato \
  --output data/manifests/somato_pilot_raw.json
python -m jacaccess.io.download finalize \
  --raw-root data/raw/kronemer \
  --output data/manifests/kronemer_pilot_raw.json

python automation/inspect_pilots.py
