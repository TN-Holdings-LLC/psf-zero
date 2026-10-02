#!/usr/bin/env bash
# run_hold_2026-10-03.sh -- runs hold_eval.py at home (WSL): 216 jobs (9 devices x 4 arms x 6 families), then scores.
#   bash run_hold_2026-10-03.sh <repo> <out dir>            scored run
#   SMOKE=1 bash run_hold_2026-10-03.sh <repo> <out dir>    plumbing and timing check
#   PAR=<n> parallel jobs (default 6); it changes only the wall time.
# Progress: grep -c "   done" <log>   (216 when all jobs are finished)
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/hold_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-6}"; SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
{
  echo "date_utc $(date -u +%FT%TZ)"
  lscpu | grep -E "Model name|^CPU\(s\)"
  python -c "import sys, qiskit, qiskit_aer; print('python', sys.version.split()[0], 'qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)"
  sha256sum "$S" "$0"
  git -C "$REPO" log --oneline -1
  echo "PAR $PAR"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
t0=$(date +%s)
run_one() {
  python -u "$S" run --repo "$REPO" --out "$OUT" --device "$1" --arm "$2" --family "$3" $SM > "$OUT/log_$1_$2_$3.txt" 2>&1
  echo "   done $1 $2 $3 ($(( $(date +%s) - t0 )) s)"
}
export -f run_one; export S REPO OUT SM t0
for a in A7 C5 C3 L3T; do for f in F1 F3 F4 F2 F6 F5; do
  for d in FakeKingston FakeFez FakeMarrakesh FakeAachen FakeTorino FakeAuckland FakeHanoiV2 FakeAlgiers FakeGeneva; do
    echo "$d $a $f"; done; done; done |
  xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "== all jobs finished after $(( $(date +%s) - t0 )) s"
grep -h "Traceback\|Error" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -n 60
echo "HOLD DONE"
