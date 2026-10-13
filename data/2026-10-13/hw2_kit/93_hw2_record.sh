#!/bin/bash
# Step 3 of HW2 (VPN ON again; no IBM connection): scores HW2 offline, hashes every result file and copies everything
# to the kit's results/ on the PC. Commits nothing (the Addenda and the commit come in the next kit).
export TZ=Europe/Zurich
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
cd "$R" || stop "repository not found"
source "$V" || stop "Python environment not found"
OUT=data/2026-10-13/hw2; LG=data/2026-10-13/hw2_logs
[ -e "$OUT/hw2_raw_counts.json" ] || stop "no HW2 results yet (run 92 first)"
timeout 1800 python -u "$S/hw2_real.py" score --out "$OUT" 2>&1 | tee "$LG/hw2_score_log.txt" | tail -40
[ -e "$OUT/hw2_score.md" ] || stop "scoring failed (see $LG/hw2_score_log.txt)"
(cd data/2026-10-13 && find hw2 hw2_logs -type f | sort | xargs sha256sum > HW2_SHA256SUMS.txt)
mkdir -p "$S/results"
cp -r "$OUT" "$LG" data/2026-10-13/HW2_SHA256SUMS.txt "$S/results/"
echo "== 93 DONE $(date '+%F %T %Z'). Everything is in the kit's results/ on the PC."
