#!/usr/bin/env bash
# run_depth_r.sh -- DEPTH-R (Addendum 399): train (4 jobs), deploy (24 jobs), finetune (4 jobs), score.
#   bash benchmarks/run_depth_r.sh <out dir>            scored run
#   DRY=1 bash benchmarks/run_depth_r.sh <out dir>      dry run (DEPTH's development seed 2, 12 points, L 1/4/12)
#   PAR=<n> parallel jobs (default 6). Run from the repository root.
set -uo pipefail
OUT="$1"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/depth_r_eval.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
PAR="${PAR:-6}"; D=""; [ "${DRY:-0}" = 1 ] && D="--dry"
{ date -u +%FT%TZ; nproc; git -C "$HERE/.." rev-parse --short HEAD; git -C "$HERE/.." status --short --untracked-files=no
  python3 -c "import sys,qiskit,qiskit_aer,sklearn,numpy;print(sys.version.split()[0],qiskit.__version__,qiskit_aer.__version__,sklearn.__version__,numpy.__version__)"
  echo "PAR $PAR DRY ${DRY:-0}"; } > "$OUT/env.txt" 2>&1
t0=$(date +%s)
job() { python3 -u "$S" "$@" --out "$OUT" $D > "$OUT/log_$(echo "$@" | tr ' -' '__').txt" 2>&1; echo "   done $* ($(( $(date +%s) - t0 )) s)"; }
export -f job; export S OUT D t0
for ds in BC D38; do for n in 4 6; do echo "train --dataset $ds --n $n"; done; done | xargs -P "$PAR" -L 1 bash -c 'job "$@"' _ > "$OUT/progress.txt"
{ for s in 1 2; do for L in 12 4; do echo "finetune --seed $s --L $L"; done; done
  for dev in FakeTorino FakeAuckland; do for a in REC RPSF L3T; do for ds in BC D38; do for n in 6 4; do
    echo "deploy --dataset $ds --n $n --device $dev --arm $a"; done; done; done; done
} | xargs -P "$PAR" -L 1 bash -c 'job "$@"' _ >> "$OUT/progress.txt"
echo "== all jobs finished after $(( $(date +%s) - t0 )) s ($(grep -c '   done' "$OUT/progress.txt") done)"
grep -l "Traceback\|STOP" "$OUT"/log_*.txt | head -5
python3 "$S" score --out "$OUT" > "$OUT/score_log.txt" 2>&1; tail -n 30 "$OUT/score_log.txt"
