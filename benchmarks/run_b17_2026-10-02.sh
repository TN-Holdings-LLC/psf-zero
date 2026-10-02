#!/usr/bin/env bash
# run_b17_2026-10-02.sh -- runs b17_practice_eval.py at home (WSL): 8 chunks (W1-W4 x n=4,6), then scores.
#   bash run_b17_2026-10-02.sh <repo> <out dir>            scored run
#   SMOKE=1 bash run_b17_2026-10-02.sh <repo> <out dir>    plumbing and timing check (a few circuits per chunk)
#   PAR=<n> parallel chunks (default 6); it changes only the wall time.
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/b17_practice_eval.py"
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
  python -u "$S" run --repo "$REPO" --out "$OUT" --workload "$1" --n "$2" $SM > "$OUT/log_$1_n$2.txt" 2>&1
  echo "   done $1 n=$2 ($(( $(date +%s) - t0 )) s)"
}
export -f run_one; export S REPO OUT SM t0
for w in W1 W2 W3 W4; do for n in 6 4; do echo "$w $n"; done; done | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
echo "== all chunks finished after $(( $(date +%s) - t0 )) s"
grep -h "Traceback\|Error" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -40
echo "B17 DONE"
