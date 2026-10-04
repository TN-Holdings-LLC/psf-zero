#!/usr/bin/env bash
# run_hold6_2026-10-04.sh -- runs hold6_eval.py at home (WSL): 486 jobs (9 devices x 4 arms x F1-F6, and 9 devices x
# 5 arms x W1-W6), then scores. The W jobs (9-10 qubits, the slow ones) are started first.
#   bash run_hold6_2026-10-04.sh <repo> <out dir>            scored run
#   SMOKE=1 bash run_hold6_2026-10-04.sh <repo> <out dir>    plumbing and timing check
#   PAR=<n> parallel jobs (default 6); it changes only the wall time.
# Progress: grep -c "   done" <out dir>/progress.txt   (486 when all jobs are finished)
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/hold6_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-6}"; SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
{
  echo "date_utc $(date -u +%FT%TZ)"
  lscpu | grep -E "Model name|^CPU\(s\)"
  python -c "import sys, qiskit, qiskit_aer; print('python', sys.version.split()[0], 'qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)"
  sha256sum "$S" "$0" "$REPO/patches/psf_compile_c11_2026-10-04/psf_compile.py" "$REPO/patches/psf_ai_compile_a8_2026-10-04/psf_ai_compile.py"
  git -C "$REPO" log --oneline -1
  echo "PAR $PAR SMOKE ${SMOKE:-0}"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
t0=$(date +%s)
run_one() {
  python -u "$S" run --repo "$REPO" --out "$OUT" --device "$1" --arm "$2" --family "$3" $SM > "$OUT/log_$1_$2_$3.txt" 2>&1
  echo "   done $1 $2 $3 ($(( $(date +%s) - t0 )) s)"
}
export -f run_one; export S REPO OUT SM t0
DEVS="FakeKingston FakeFez FakeMarrakesh FakeAachen FakeTorino FakeAuckland FakeHanoiV2 FakeAlgiers FakeGeneva"
{
  for f in W3 W4 W1 W2 W5 W6; do for a in A7 A8 C11 R3 L3T; do for d in $DEVS; do echo "$d $a $f"; done; done; done
  for f in F1 F3 F4 F2 F6 F5; do for a in A7 C11 R3 L3T; do for d in $DEVS; do echo "$d $a $f"; done; done; done
} | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _ | tee "$OUT/progress.txt" | grep -c "   done" > /dev/null
echo "== all jobs finished after $(( $(date +%s) - t0 )) s ($(grep -c "   done" "$OUT/progress.txt") done)"
grep -l "Traceback\|STOP" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -n 25
echo "HOLD6 DONE"
