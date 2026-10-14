#!/bin/bash
# Step 3 of TIME1 (VPN ON again; no IBM connection): scores TIME1 offline, hashes every result file and copies
# everything to the kit's results/ on the PC. Commits nothing (the Addendum and the commit come afterwards).
export TZ=Europe/Zurich
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
cd "$R" || stop "repository not found"
source "$V" || stop "Python environment not found"
OUT=data/2026-10-14/time1_run; LG=data/2026-10-14/time1_logs
[ -e "$OUT/time1_raw_counts.json" ] || stop "no TIME1 results yet (run 62 first)"
timeout 1800 python -u "$S/time1_real.py" score --out "$OUT" 2>&1 | tee "$LG/time1_score_log.txt" | tail -14
[ -e "$OUT/time1_score.md" ] || stop "scoring failed (see $LG/time1_score_log.txt)"
(cd data/2026-10-14 && find time1_run time1_logs -type f | sort | xargs sha256sum > TIME1_SHA256SUMS.txt)
mkdir -p "$S/results"
cp -r "$OUT" "$LG" data/2026-10-14/TIME1_SHA256SUMS.txt "$S/results/"
echo "== 63 DONE $(date '+%F %T %Z'). Everything is in the kit's results/ on the PC."
