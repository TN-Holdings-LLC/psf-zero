#!/usr/bin/env bash
# run_readout.sh <out> <c12 psf_compile.py> <c13 psf_compile.py>   [DRY=1] [PAR=2]
set -uo pipefail
OUT="$1"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
abs() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
C12="$(abs "$2")"; C13="$(abs "$3")"; HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/readout_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-2}"; D=""; [ "${DRY:-0}" = 1 ] && D="--dry"
{ date -u +%FT%TZ; python3 -c "import sys,qiskit,qiskit_aer;print(sys.version.split()[0],qiskit.__version__,qiskit_aer.__version__)"
  sha256sum "$S" "$HERE/depth_eval.py" "$0" "$C12" "$C13"; echo "PAR $PAR DRY ${DRY:-0}"; } > "$OUT/env.txt" 2>&1
t0=$(date +%s)
python3 -u "$S" train --out "$OUT" $D > "$OUT/log_train.txt" 2>&1
echo "train done ($(( $(date +%s) - t0 )) s)" > "$OUT/progress.txt"
for d in FakeTorino FakeKingston FakeAuckland; do echo $d; done | xargs -P "$PAR" -I{} bash -c \
  "python3 -u '$S' run --device {} --out '$OUT' --c12 '$C12' --c13 '$C13' $D > '$OUT/log_{}.txt' 2>&1; echo '   done {} '\$(( \$(date +%s) - $t0 ))' s' >> '$OUT/progress.txt'"
grep -l "Traceback" "$OUT"/log_*.txt | head
python3 "$S" score --out "$OUT" > "$OUT/score_log.txt" 2>&1; tail -n 25 "$OUT/score_log.txt"
