#!/bin/bash
# Step 3 of HW3 (VPN ON again; no IBM connection): scores HW3 offline, hashes every result file and copies everything
# to the kit's results/ on the PC. Commits nothing (the Addenda and the commit come in the next kit).
export TZ=Europe/Zurich
D=${HW3_DATE:-$(date +%F)}
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
cd "$R" || stop "repository not found"
source "$V" || stop "Python environment not found"
OUT=data/$D/hw3; LG=data/$D/hw3_logs
[ -e "$OUT/hw3_raw_counts.json" ] || stop "no HW3 results yet (run 96 first)"
timeout 1800 python -u "$S/hw3_real.py" score --out "$OUT" 2>&1 | tee "$LG/hw3_score_log.txt" | tail -40
[ -e "$OUT/hw3_score.md" ] || stop "scoring failed (see $LG/hw3_score_log.txt)"
(cd data/$D && find hw3 hw3_logs -type f | sort | xargs sha256sum > HW3_SHA256SUMS.txt)
mkdir -p "$S/results"
cp -r "$OUT" "$LG" data/$D/HW3_SHA256SUMS.txt "$S/results/"
echo "== 97 DONE $(date '+%F %T %Z'). Everything is in the kit's results/ on the PC."
