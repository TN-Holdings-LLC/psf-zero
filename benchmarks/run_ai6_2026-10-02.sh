#!/usr/bin/env bash
# run_ai6_2026-10-02.sh -- runs ai6_eval.py at home (WSL): prep (model circuits), 90 jobs (3 devices x 5 arms x 6 sets),
# then scores.
#   bash run_ai6_2026-10-02.sh <repo> <out dir>            scored run
#   SMOKE=1 bash run_ai6_2026-10-02.sh <repo> <out dir>    plumbing and timing check
#   PAR=<n> parallel jobs (default 6); it changes only the wall time.
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/ai6_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-6}"; SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
{
  echo "date_utc $(date -u +%FT%TZ)"
  lscpu | grep -E "Model name|^CPU\(s\)"
  python -c "import sys, qiskit, qiskit_aer, pennylane; print('python', sys.version.split()[0], 'qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__, 'pennylane', pennylane.__version__)"
  sha256sum "$S" "$0"
  git -C "$REPO" log --oneline -1
  echo "PAR $PAR"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
python -u "$S" prep --repo "$REPO" --out "$OUT" $SM 2>&1 | tee "$OUT/log_prep.txt" | tail -n 3
t0=$(date +%s)
run_one() {
  python -u "$S" run --repo "$REPO" --out "$OUT" --device "$1" --arm "$2" --set "$3" $SM > "$OUT/log_$1_$2_$3.txt" 2>&1
  echo "   done $1 $2 $3 ($(( $(date +%s) - t0 )) s)"
}
export -f run_one; export S REPO OUT SM t0
for a in A6 A5 A6F C5 L3T; do for s in F1 F4 F3 F2 F5 MODEL; do for d in FakeKingston FakeTorino FakeAuckland; do echo "$d $a $s"; done; done; done |
  xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "== all jobs finished after $(( $(date +%s) - t0 )) s"
grep -h "Traceback\|Error" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -n 32
echo "AI6 DONE"
