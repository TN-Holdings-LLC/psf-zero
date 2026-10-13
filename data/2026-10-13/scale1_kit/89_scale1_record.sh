#!/bin/bash
# Step 3 of SCALE1 (VPN ON again; no IBM connection): scores SCALE1 offline, hashes every result file and copies
# everything to the kit's results/ on the PC. Commits nothing (the Addendum and the commit come in the next kit).
export TZ=Europe/Zurich
D=${SCALE1_DATE:-$(date +%F)}
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
cd "$R" || stop "repository not found"
source "$V" || stop "Python environment not found"
OUT=data/$D/scale1; LG=data/$D/scale1_logs
[ -e "$OUT/scale1_raw_counts_main.json" ] || stop "no SCALE1 results yet (run 88 first)"
timeout 1800 python -u "$S/scale1_real.py" score --out "$OUT" 2>&1 | tee "$LG/scale1_score_log.txt" | tail -25
[ -e "$OUT/scale1_score.md" ] || stop "scoring failed (see $LG/scale1_score_log.txt)"
(cd data/$D && find scale1 scale1_logs -type f | sort | xargs sha256sum > SCALE1_SHA256SUMS.txt)
mkdir -p "$S/results"
cp -r "$OUT" "$LG" data/$D/SCALE1_SHA256SUMS.txt "$S/results/"
echo "== 89 DONE $(date '+%F %T %Z'). Everything is in the kit's results/ on the PC."
