#!/usr/bin/env bash
# run_exact_2026-10-05.sh -- runs exact_eval.py at home (WSL): 9 part-Y jobs (started first, the slow ones) and 36
# part-X jobs, then scores.
#   bash run_exact_2026-10-05.sh <repo> <out dir>            scored run
#   SMOKE=1 bash run_exact_2026-10-05.sh <repo> <out dir>    plumbing and timing check
#   PAR=<n> parallel jobs (default 6); it changes only the wall time.
# Progress: grep -c "   done" <out dir>/progress.txt   (45 when all jobs are finished)
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/exact_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-6}"; SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
{
  echo "date_utc $(date -u +%FT%TZ)"
  lscpu | grep -E "Model name|^CPU\(s\)"
  python -c "import sys, qiskit, qiskit_aer; print('python', sys.version.split()[0], 'qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)"
  sha256sum "$S" "$0" "$REPO/patches/psf_compile_c12_2026-10-05/psf_compile.py" "$REPO/patches/psf_ai_compile_a9_2026-10-05/psf_ai_compile.py"
  git -C "$REPO" log --oneline -1
  echo "PAR $PAR SMOKE ${SMOKE:-0}"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
t0=$(date +%s)
run_one() {   # x <device> <arm>  |  y <device> -
  if [ "$1" = x ]; then
    python -u "$S" x --repo "$REPO" --out "$OUT" --device "$2" --arm "$3" $SM > "$OUT/log_x_$2_$3.txt" 2>&1
  else
    python -u "$S" y --repo "$REPO" --out "$OUT" --device "$2" $SM > "$OUT/log_y_$2.txt" 2>&1
  fi
  echo "   done $1 $2 $3 ($(( $(date +%s) - t0 )) s)"
}
export -f run_one; export S REPO OUT SM t0
{
  for d in FakeKingston FakeFez FakeMarrakesh FakeAachen FakeTorino FakeAuckland FakeHanoiV2 FakeAlgiers FakeGeneva; do echo "y $d -"; done
  for a in A8 A9 R41 C12 RPSF L3T; do
    for d in FakeAuckland FakeHanoiV2 FakeAlgiers FakeGeneva FakeTorino FakeKingston; do echo "x $d $a"; done
  done
} | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _ > "$OUT/progress.txt"
echo "== all jobs finished after $(( $(date +%s) - t0 )) s ($(grep -c "   done" "$OUT/progress.txt") done)"
grep -l "Traceback\|STOP" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -n 22
echo "EXACT DONE"
