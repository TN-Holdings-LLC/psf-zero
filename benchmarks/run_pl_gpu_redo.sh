#!/bin/bash
# PL-GPU-REDO runner (Addendum 375). Run from the repository root inside the venv:
#   bash benchmarks/run_pl_gpu_redo.sh OUT_DIR             scored run: 4 arms x 2 spares x 30 laps, one arm at a time
#   bash <kit>/benchmarks/run_pl_gpu_redo.sh OUT_DIR smoke  smoke: 2 laps per spare, flagged as smoke
set -u
OUT=$1
MODE=${2:-scored}
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
EXTRA=""
[ "$MODE" = "smoke" ] && EXTRA="--laps 2 --smoke"
{
  echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ) mode $MODE"
  echo "head $(git rev-parse --short HEAD)"
  git status --short --untracked-files=no
  uname -sr
  nproc
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python -c "import sys, qiskit, qiskit_aer, pennylane, numpy, psf_zero_core as c; print(sys.version.split()[0], qiskit.__version__, qiskit_aer.__version__, pennylane.__version__, numpy.__version__, getattr(c, 'CORE_VERSION', None))"
} > "$OUT/env.txt" 2>&1
for a in R RR A12 Q3; do
  echo "$(date -u +%H:%M:%S) start $a" >> "$OUT/progress.txt"
  python -u "$S/pl_gpu_redo.py" run --arm "$a" --out "$OUT" $EXTRA > "$OUT/log_$a.txt" 2>&1
  echo "$(date -u +%H:%M:%S) end $a rc $?" >> "$OUT/progress.txt"
  grep -E "C0 GPU vs CPU" "$OUT/log_$a.txt" | head -2
  grep -E " lap +[0-9]+ compile" "$OUT/log_$a.txt" | tail -4
  grep -E "Traceback|Error" "$OUT/log_$a.txt" | head -3
done
if [ "$MODE" = "smoke" ]; then
  echo "smoke done $(date -u +%H:%M:%S)"
else
  python -u "$S/pl_gpu_redo.py" score --out "$OUT" > "$OUT/score_log.txt" 2>&1
  python -u "$S/pl_gpu_redo_verify.py" "$OUT" > "$OUT/verify_log.txt" 2>&1
  echo "$(date -u +%H:%M:%S) scored and verified" >> "$OUT/progress.txt"
  cat "$OUT/verify_log.txt"
fi
