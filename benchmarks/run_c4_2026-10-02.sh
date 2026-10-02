#!/usr/bin/env bash
# run_c4_2026-10-02.sh -- runs c4_eval.py at home (WSL): 45 jobs (3 devices x 3 arms x 5 families), then scores.
#   bash run_c4_2026-10-02.sh <repo> <out dir>            scored run
#   SMOKE=1 bash run_c4_2026-10-02.sh <repo> <out dir>    plumbing and timing check
#   PAR=<n> parallel jobs (default 6); it changes only the wall time.
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/c4_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-6}"; SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
{
  echo "date_utc $(date -u +%FT%TZ)"
  lscpu | grep -E "Model name|^CPU\(s\)"
  python -c "import sys; print('python', sys.version.split()[0])"
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
for a in C4 C3 L3T; do for d in FakeKingston FakeTorino FakeAuckland; do for f in F1 F2 F3 F4 F5; do echo "$d $a $f"; done; done; done |
  xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "== all jobs finished after $(( $(date +%s) - t0 )) s"
grep -h "Traceback\|Error" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -30
echo "C4 DONE"
