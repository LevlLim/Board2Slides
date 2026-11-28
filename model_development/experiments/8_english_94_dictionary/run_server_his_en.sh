#!/usr/bin/env bash
# Final-model recipe with the 94-character English dictionary (b2n_server_his_en.py).
#   bash run_server_his_en.sh --link-data-from mobile_official_20261110
# Uses the persistent environment and PaddleOCR checkout that run_runpod_mobile.sh set up, and the same volume lock,
# so it never runs at the same time as another Board2Notes controller.
set -euo pipefail
PKG="$(cd "$(dirname "$0")" && pwd -P)"
VOL="${B2N_STORAGE_ROOT:-/workspace}"
export B2N_WORK="$VOL/b2n"
[[ -x "$VOL/venv_b2n/bin/python" && -f "$VOL/venv_b2n/.b2n_installed" ]] || {
  echo "The persistent environment is missing: run run_runpod_mobile.sh --setup-only first." >&2; exit 1; }
exec 9>"$B2N_WORK/controller.lock"
flock -n 9 || { echo "Another Board2Notes controller holds this volume's lock; not starting." >&2; exit 1; }
# shellcheck disable=SC1091
source "$VOL/venv_b2n/bin/activate"
export PIP_CACHE_DIR="$VOL/.pip_cache" HF_HOME="$VOL/.hf_cache" XDG_CACHE_HOME="$VOL/.cache" PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
cd "$PKG"
for Z in b2n_versions_data.zip b2n_newwriter_data.zip; do
  if [[ ! -s "$PKG/$Z" ]]; then
    for C in "$VOL/board2notes_versions_runpod/$Z" "$VOL/b2n/$Z"; do [[ -s "$C" ]] && { ln -s "$C" "$PKG/$Z"; break; }; done
  fi
  [[ -s "$PKG/$Z" ]] || { echo "Missing $Z (it is in the server package on the volume)." >&2; exit 1; }
done
python -u b2n_server_his_en.py all "$@"
echo "DONE. Download results_<experiment>.zip from $PKG."
