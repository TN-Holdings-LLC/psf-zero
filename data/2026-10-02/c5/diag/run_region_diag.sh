#!/usr/bin/env bash
# run_region_diag.sh -- exploratory diagnosis for Addendum 308 (not a test): 3 devices in parallel, then the summary.
#   bash run_region_diag.sh <repo> <out dir>
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
S="$(cd "$(dirname "$0")" && pwd)/region_diag.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
{ echo "date_utc $(date -u +%FT%TZ)"; python -c "import sys, qiskit, qiskit_aer; print('python', sys.version.split()[0], 'qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)"; git -C "$REPO" log --oneline -1; } > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
t0=$(date +%s)
for d in FakeAuckland FakeTorino FakeKingston; do
  python -u "$S" run --repo "$REPO" --out "$OUT" --device "$d" > "$OUT/log_$d.txt" 2>&1 &
done
wait
echo "== runs finished after $(( $(date +%s) - t0 )) s"
tail -n 2 "$OUT"/log_*.txt
python "$S" summary --out "$OUT" > /dev/null
echo "DIAG DONE"
