#!/usr/bin/env bash
# run_qml_home2_2026-10-02.sh -- runs qml_home2_eval.py at home (WSL), then scores it.
#   bash run_qml_home2_2026-10-02.sh <repo> <out dir>            scored run: Q1D, then 96 Q2W runs in parallel
#   SMOKE=1 bash run_qml_home2_2026-10-02.sh <repo> <out dir>    plumbing and timing check (other seeds, 2 steps)
#   PAR=<n> sets the number of parallel Q2W processes (default 6). It changes only the wall time.
# Every process is single-threaded so that parallel runs do not compete for cores.
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/qml_home2_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-6}"
SM=""; SEEDS="41 42 43 44 45 46 47 48"
if [ "${SMOKE:-0}" = 1 ]; then SM="--smoke"; SEEDS="941"; fi
DEVS="FakeAuckland FakeTorino FakeKingston"
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
echo "== Q1D $(date -u +%T)"
python -u "$S" q1 --repo "$REPO" --out "$OUT" $SM > "$OUT/log_q1.txt" 2>&1 &
Q1PID=$!
echo "== Q2W $(date -u +%T)"
run_one() {
  python -u "$S" q2 --repo "$REPO" --out "$OUT" --arm "$1" --device "$2" --seed "$3" $SM > "$OUT/log_q2_$1_$2_$3.txt" 2>&1
  echo "   done $1 $2 $3 ($(( $(date +%s) - t0 )) s)"
}
export -f run_one; export S REPO OUT SM t0
for d in $DEVS; do for a in REL C2 A5 L3T; do for s in $SEEDS; do echo "$a $d $s"; done; done; done |
  xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
wait $Q1PID
echo "== all runs finished after $(( $(date +%s) - t0 )) s"
grep -h "STOP\|Traceback\|Error" "$OUT"/log_*.txt | head -5
python "$S" score --out "$OUT" | tee "$OUT/score_log.txt" | tail -40
echo "QML HOME2 DONE"
