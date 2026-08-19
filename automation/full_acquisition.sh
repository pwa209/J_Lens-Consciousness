#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
cd "$ROOT"
source .venv/bin/activate
mkdir -p data/downloads/full/kronemer data/raw/gabor data/raw/kronemer data/manifests

check_disk() {
  local free_kib
  free_kib=$(df --output=avail -k /root/autodl-tmp | tail -1 | tr -d ' ')
  if (( free_kib < 524288000 )); then
    echo "Refusing acquisition/extraction: less than 500 GiB scratch remains" >&2
    return 70
  fi
}

check_disk
Kronemer_manifest="configs/data_sources/kronemer_downloads.tsv"
if [[ ! -s "$Kronemer_manifest" ]]; then
  python automation/build_kronemer_manifest.py \
    --output "$Kronemer_manifest"
else
  echo "Reusing frozen Kronemer manifest: $Kronemer_manifest"
fi

for archive in 223_RP_EEG.tar 238_NRP.tar; do
  pilot="data/downloads/pilots/kronemer/$archive"
  full="data/downloads/full/kronemer/$archive"
  if [[ -f "$pilot" && ! -e "$full" ]]; then
    ln "$pilot" "$full"
  fi
done

if [[ ! -f data/raw/gabor/.full_acquisition_complete ]]; then
  aws s3 sync --no-sign-request \
    s3://openneuro.org/ds005273/ \
    data/raw/gabor/
  touch data/raw/gabor/.full_acquisition_complete
fi

python automation/run_with_progress_watchdog.py \
  --watch-path data/downloads/full/kronemer \
  --watch-path data/raw/kronemer \
  --stall-seconds 600 \
  --poll-seconds 30 \
  -- \
  python automation/acquire_kronemer_hybrid.py \
    --manifest "$Kronemer_manifest" \
    --download-root data/downloads/full \
    --raw-root data/raw/kronemer \
    --inventory-root results/source-inspection/inventories \
    --index-root results/source-inspection/kronemer-remote-indexes \
    --archive-workers 4 \
    --member-workers 4 \
    --timeout 60 \
    --retries 6 \
    --minimum-free-gib 500

python -m jacaccess.io.download finalize \
  --raw-root data/raw/gabor \
  --output data/manifests/gabor_raw.json
python -m jacaccess.io.download finalize \
  --raw-root data/raw/somato \
  --output data/manifests/somato_raw.json
python -m jacaccess.io.download finalize \
  --raw-root data/raw/kronemer \
  --output data/manifests/kronemer_raw.json
python -m jacaccess.io.download finalize \
  --raw-root data/downloads/full/kronemer \
  --output data/manifests/kronemer_archives.json

python automation/build_full_participant_roster.py \
  --raw-root data/raw \
  --output configs/execution/participants.tsv

touch data/manifests/FULL_ACQUISITION_COMPLETE
