#!/bin/bash
# HW2, extra step for NS2 (VPN OFF, right after 92_hw2_run.sh; read-only, no job, no QPU time): saves the
# calibration of HW2's device (named in HW2's manifest) to data/2026-10-13/hw2/. Runs no git and no pip.
export TZ=Europe/Zurich
git() { echo "STOP: git is not used in this step"; exit 1; }
pip() { echo "STOP: pip is not used in this step"; exit 1; }
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
cd "$R" || { echo "STOP: repository not found"; exit 1; }
source "$V" || { echo "STOP: Python environment not found"; exit 1; }
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || { echo "STOP: the kit's files differ from SHA256SUMS"; exit 1; }
[ -e data/2026-10-13/hw2/hw2_manifest.json ] || { echo "STOP: no HW2 manifest yet (run 92 first)"; exit 1; }
timeout 600 python -u "$S/ns_snapshot.py" data/2026-10-13/hw2 2>&1 | tail -2
ls data/2026-10-13/hw2/ns_snapshot_*.json > /dev/null 2>&1 || { echo "STOP: no snapshot written"; exit 1; }
mkdir -p "$S/results"; cp data/2026-10-13/hw2/ns_*.json "$S/results/"
echo "== 92s DONE $(date '+%F %T %Z'). Turn the VPN back ON, then run 93 as planned."
