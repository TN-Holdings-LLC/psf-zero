#!/usr/bin/env bash
# run_depth.sh -- QML-2 DEPTH stage 1: train (4 jobs), deploy (24 jobs), finetune (4 jobs), score.
#   bash run_depth.sh <out dir> <path to candidate psf_compile.py>          scored run
#   DRY=1 bash run_depth.sh <out dir> <path>                                 dry run (seed 2, small)
#   PAR=<n> parallel jobs (default 2); needs PYTHONPATH with the Rust core (psf_zero_core).
set -uo pipefail
OUT="$1"; C12="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/depth_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-2}"; D=""; [ "${DRY:-0}" = 1 ] && D="--dry"
{ date -u +%FT%TZ; nproc; python3 -c "import sys,qiskit,qiskit_aer,sklearn;print(sys.version.split()[0],qiskit.__version__,qiskit_aer.__version__,sklearn.__version__)"
  sha256sum "$S" "$0" "$C12"; echo "PAR $PAR DRY ${DRY:-0}"; } > "$OUT/env.txt" 2>&1
t0=$(date +%s)
job() { python3 -u "$S" "$@" --out "$OUT" --c12 "$C12" $D > "$OUT/log_$(echo "$@" | tr ' -' '__').txt" 2>&1; echo "   done $* ($(( $(date +%s) - t0 )) s)"; }
export -f job; export S OUT C12 D t0
for ds in BC D38; do for n in 4 6; do echo "train --dataset $ds --n $n"; done; done | xargs -P "$PAR" -L 1 bash -c 'job "$@"' _ > "$OUT/progress.txt"
{ for s in 1 2; do for L in 12 4; do echo "finetune --seed $s --L $L"; done; done
  for dev in FakeTorino FakeAuckland; do for a in C12 RPSF L3T; do for ds in BC D38; do for n in 6 4; do
    echo "deploy --dataset $ds --n $n --device $dev --arm $a"; done; done; done; done
} | xargs -P "$PAR" -L 1 bash -c 'job "$@"' _ >> "$OUT/progress.txt"
echo "== all jobs finished after $(( $(date +%s) - t0 )) s ($(grep -c '   done' "$OUT/progress.txt") done)"
grep -l "Traceback\|STOP" "$OUT"/log_*.txt | head -5
python3 "$S" score --out "$OUT" > "$OUT/score_log.txt" 2>&1; tail -n 25 "$OUT/score_log.txt"
