#!/usr/bin/env bash
# run_recr_2026-10-06.sh -- runs recr_eval.py at home (WSL): one job per device (6), then scores.
#   bash benchmarks/run_recr_2026-10-06.sh <repo> <out dir>            scored run
#   SMOKE=1 bash benchmarks/run_recr_2026-10-06.sh <repo> <out dir>    plumbing check
#   PAR=<n> parallel jobs (default 6); it changes only the wall time.
# Progress: grep -c "   done" <out dir>/progress.txt   (6 when all jobs are finished)
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
S="$REPO/benchmarks/recr_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-6}"; SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
{
  echo "date_utc $(date -u +%FT%TZ)"
  lscpu | grep -E "Model name|^CPU\(s\)"
  python -c "import sys, numpy, qiskit, qiskit_aer; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)"
  sha256sum "$S" "$0" "$REPO/patches/psf_compile_c14_2026-10-05/psf_compile.py" "$REPO/patches/psf_ai_compile_a11_2026-10-05/psf_ai_compile.py"
  git -C "$REPO" log --oneline -1
  echo "PAR $PAR SMOKE ${SMOKE:-0}"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
t0=$(date +%s)
run_one() {
  python -u "$S" run --device "$1" --out "$OUT" $SM > "$OUT/log_$1.txt" 2>&1
  echo "   done $1 ($(( $(date +%s) - t0 )) s)"
}
export -f run_one; export S OUT SM t0
printf "%s\n" FakeKingston FakeTorino FakeBrussels FakeOsaka FakeHanoiV2 FakeAuckland \
  | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _ > "$OUT/progress.txt"
echo "== all jobs finished after $(( $(date +%s) - t0 )) s ($(grep -c "   done" "$OUT/progress.txt") done)"
grep -l "Traceback\|STOP" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -n 16
echo "RECR DONE"
