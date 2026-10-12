#!/bin/bash
# Step 3 of the real-device session (VPN ON again; no IBM connection): scores HW1 and PSF-XOR offline, hashes every
# result file and copies everything to the kit's results/ on the PC. Commits nothing (the Addenda and the commit come
# in the next kit, after the results are read).
export TZ=Europe/Zurich
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
cd "$R" || stop "repository not found"
source "$V" || stop "Python environment not found"
OUT=data/2026-10-12/hw1; LG=data/2026-10-12/hw1_logs; ST=data/2026-10-12/stage2
[ -e "$OUT/hw1_raw_counts.json" ] || stop "no HW1 results yet (run 82 first)"
timeout 1800 python -u "$S/hw1_real.py" score --out "$OUT" 2>&1 | tee "$LG/hw1_score_log.txt" | tail -60
[ -e "$OUT/hw1_score.md" ] || stop "scoring failed (see $LG/hw1_score_log.txt)"
cp stage2_manifest_2026-09-28*.json "$ST/" 2>/dev/null
(cd data/2026-10-12 && find stage2 hw1 hw1_logs -type f | sort | xargs sha256sum > HW_SHA256SUMS.txt)
mkdir -p "$S/results"
cp -r "$ST" "$OUT" "$LG" data/2026-10-12/HW_SHA256SUMS.txt "$S/results/"
echo "== Stage 2"; grep -E "S[1-4] |P0 " "$ST/stage2_score.txt"
echo "== 83 DONE $(date '+%F %T %Z'). Everything is in the kit's results/ on the PC."
