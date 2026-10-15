#!/bin/bash
# Step 3 of RING1 (VPN ON again; no IBM connection): scores RING1 offline, hashes every result file and copies
# everything to the kit's results/ on the PC. Commits nothing (the Addendum and the commit come afterwards).
export TZ=Europe/Zurich
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
cd "$R" || stop "repository not found"
source "$V" || stop "Python environment not found"
OUT=data/2026-10-15/ring1_run; LG=data/2026-10-15/ring1_logs
[ -e "$OUT/ring1_raw_counts.json" ] || stop "no RING1 results yet (run 68 first)"
timeout 1800 python -u "$S/ring1_real.py" score --out "$OUT" 2>&1 | tee "$LG/ring1_score_log.txt" | tail -24
[ -e "$OUT/ring1_score.md" ] || stop "scoring failed (see $LG/ring1_score_log.txt)"
(cd data/2026-10-15 && find ring1_run ring1_logs -type f | sort | xargs sha256sum > RING1_SHA256SUMS.txt)
mkdir -p "$S/results"
cp -r "$OUT" "$LG" data/2026-10-15/RING1_SHA256SUMS.txt "$S/results/"
echo "== 69 DONE $(date '+%F %T %Z'). Everything is in the kit's results/ on the PC."
