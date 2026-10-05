#!/usr/bin/env bash
# run_ai10b.sh <out>  [DRY=1] [PAR=2]   (run in this folder's copy; PYTHONPATH with the Rust core)
set -uo pipefail
OUT="$1"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"; HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/ai10_eval2.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-2}"; D=""; [ "${DRY:-0}" = 1 ] && D="--dry"
{ date -u +%FT%TZ; sha256sum "$S" "$0" "$HERE"/psf_ai_compile.py "$HERE"/psf_ai_compile_a9.py "$HERE"/psf_compile.py "$HERE"/depth_eval.py "$HERE"/readout_eval.py; echo "PAR $PAR DRY ${DRY:-0}"; } > "$OUT/env.txt"
t0=$(date +%s)
for d in FakeTorino FakeKingston FakeAuckland; do echo $d; done | xargs -P "$PAR" -I{} bash -c \
  "python3 -u '$S' run --device {} --out '$OUT' $D > '$OUT/log_{}.txt' 2>&1; echo '   done {} '\$(( \$(date +%s) - $t0 ))' s' >> '$OUT/progress.txt'"
grep -l Traceback "$OUT"/log_*.txt | head
python3 "$S" score --out "$OUT" > "$OUT/score_log.txt" 2>&1; tail -n 20 "$OUT/score_log.txt"
