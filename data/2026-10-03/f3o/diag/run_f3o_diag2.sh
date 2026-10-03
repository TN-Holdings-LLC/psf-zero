#!/usr/bin/env bash
# run_f3o_diag2.sh -- exploratory diagnosis, part 2 (not a test): 5 devices in parallel, then the summary.
#   bash run_f3o_diag2.sh <repo> <out dir>
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
S="$(cd "$(dirname "$0")" && pwd)/f3o_diag2.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
{ echo "date_utc $(date -u +%FT%TZ)"; git -C "$REPO" log --oneline -1; } > "$OUT/env.txt"
for d in FakeAuckland FakeHanoiV2 FakeAlgiers FakeGeneva FakeTorino; do
  python -u "$S" run --repo "$REPO" --out "$OUT" --device "$d" > "$OUT/log_$d.txt" 2>&1 &
done
wait
tail -n 1 "$OUT"/log_*.txt
python "$S" summary --out "$OUT"
echo "F3O DIAG2 DONE"
