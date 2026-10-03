#!/usr/bin/env bash
# run_a7gap_diag.sh -- exploratory diagnosis of the remaining gap between release 2026-10-03.2 and a7 (not a test):
# 5 devices in parallel, then the summary.
#   bash run_a7gap_diag.sh <repo> <out dir>
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
S="$(cd "$(dirname "$0")" && pwd)/a7gap_diag.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
{ echo "date_utc $(date -u +%FT%TZ)"; git -C "$REPO" log --oneline -1; } > "$OUT/env.txt"
for d in FakeAuckland FakeGeneva FakeAlgiers FakeHanoiV2 FakeTorino FakeMarrakesh; do
  python -u "$S" run --repo "$REPO" --out "$OUT" --device "$d" > "$OUT/log_$d.txt" 2>&1 &
done
wait
tail -n 1 "$OUT"/log_*.txt
python "$S" summary --out "$OUT"
echo "A7GAP DIAG DONE"
