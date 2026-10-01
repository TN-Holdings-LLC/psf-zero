#!/usr/bin/env bash
# run_qml_home_2026-10-01.sh -- runs qml_home_eval.py at home (WSL), then scores it.
#   bash run_qml_home_2026-10-01.sh <repo> <out dir>             scored run: Q1, then 16 Q2 runs, 6 in parallel
#   SMOKE=1 bash run_qml_home_2026-10-01.sh <repo> <out dir>     plumbing check (other data seeds, 2 SPSA steps)
# Every process is single-threaded so that parallel runs do not compete for cores.
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/qml_home_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
SM=""; SEEDS="41 42"; DEVS="FakeAuckland FakeTorino"
if [ "${SMOKE:-0}" = 1 ]; then SM="--smoke"; SEEDS="941"; DEVS="FakeAuckland"; fi
{
  echo "date_utc $(date -u +%FT%TZ)"
  lscpu | grep -E "Model name|^CPU\(s\)"
  python -c "import sys; print('python', sys.version.split()[0])"
  sha256sum "$S" "$0"
  git -C "$REPO" log --oneline -1
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
t0=$(date +%s)
echo "== Q1 $(date -u +%T)"
python -u "$S" q1 --repo "$REPO" --out "$OUT" $SM > "$OUT/log_q1.txt" 2>&1 &
Q1PID=$!
echo "== Q2 $(date -u +%T)"
run_one() {
  python -u "$S" q2 --repo "$REPO" --out "$OUT" --arm "$1" --device "$2" --seed "$3" $SM > "$OUT/log_q2_$1_$2_$3.txt" 2>&1
  echo "   done $1 $2 $3 ($(( $(date +%s) - t0 )) s)"
}
export -f run_one; export S REPO OUT SM t0
for d in $DEVS; do for a in REL C2 A5 L3T; do for s in $SEEDS; do echo "$a $d $s"; done; done; done |
  xargs -P 6 -L 1 bash -c 'run_one "$@"' _
wait $Q1PID
echo "== Q1 finished; all runs finished after $(( $(date +%s) - t0 )) s"
grep -h "STOP\|Traceback\|Error" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -30
echo "QML HOME DONE"
