#!/bin/bash
# C63-VAL, the rest, at home: part R's 9 remaining tests and part D, with the workplace's script and criteria
# (pre-registration fixed 2026-10-15 08:52:23 CEST). Then the score over the workplace's 20 cells and these.
# Speed (S1, S2) measured here is from another machine than the workplace's and is reported as such; the correctness
# predictions do not depend on the machine. No network. Expected: about 1.5 hours.
# Never overwrites: refuses an existing output folder.
set -u
PSF=~/psf_zero_fresh_test
DAY=/mnt/c/Users/Administrator/Desktop/psf_zero_project/day_2026-10-15
W=data/2026-10-15/workplace/workplace_2026-10-15c_W1-W3/w34
OUT="$DAY/c63val_home"
cd "$PSF" || { echo "STOP: no $PSF"; exit 1; }
[ -f "$W/c63_val.py" ] || { echo "STOP: $W not in this clone (pull first)"; exit 1; }
[ -e "$OUT" ] && { echo "STOP: $OUT exists; a run never overwrites another"; exit 1; }
mkdir -p "$OUT" && cp -r "$W/run63" "$OUT/run63"
echo "start $(TZ=Europe/Zurich date '+%Y-%m-%d %H:%M:%S %Z')" | tee "$OUT/times.txt"
OMP_NUM_THREADS=1 C63_ONLY=hhtree_heis_n16,ghz_n12,chain_cz_n12,chain_cz_n16,ring_cz_n12,hh_cz_n16,hh_cz_n20,cx_chain_n12,cx_chain_n16 \
  python "$W/c63_val.py" "$OUT/R2" R > "$OUT/R2.log" 2>&1 || { echo "STOP: part R failed, see $OUT/R2.log"; exit 1; }
OMP_NUM_THREADS=1 python "$W/c63_val.py" "$OUT/D" D > "$OUT/D.log" 2>&1 || { echo "STOP: part D failed, see $OUT/D.log"; exit 1; }
cp "$OUT/R2/c63_val_rows_R.jsonl" "$OUT/run63/R/c63_val_rows_R2_home.jsonl"
mkdir -p "$OUT/run63/D" && cp "$OUT"/D/c63_val_rows_*.jsonl "$OUT/run63/D/"
python "$W/c63_score.py" "$OUT/run63" | tee "$OUT/c63_val_score_home.txt"
echo "end $(TZ=Europe/Zurich date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$OUT/times.txt"
python -m zipfile -c "$DAY/c63val_home.zip" "$OUT" && echo "== DONE: $DAY/c63val_home.zip"
