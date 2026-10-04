#!/usr/bin/env bash
# run_h4_diag.sh -- exploratory diagnosis of HOLD6's H4 miss (FakeAlgiers 4-qubit GHZ chains; not a test): the detail
# part, then the nine devices' rescore in parallel, then the summary.
#   bash run_h4_diag.sh <repo> <out dir>        PAR=<n> (default 6)
set -uo pipefail
REPO="$(cd "$1" && pwd)"; OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
S="$(cd "$(dirname "$0")" && pwd)/h4_diag.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
export S REPO OUT
PAR="${PAR:-6}"
{ echo "date_utc $(date -u +%FT%TZ)"; sha256sum "$S" "$0"; git -C "$REPO" log --oneline -1; echo "PAR $PAR"; } > "$OUT/env.txt"
cat "$OUT/env.txt"
t0=$(date +%s)
python -u "$S" detail --repo "$REPO" --out "$OUT" > "$OUT/log_detail.txt" 2>&1 &
for d in FakeKingston FakeFez FakeMarrakesh FakeAachen FakeTorino FakeAuckland FakeHanoiV2 FakeAlgiers FakeGeneva; do echo "$d"; done |
  xargs -P "$PAR" -I{} bash -c 'python -u "$S" rescore --repo "$REPO" --out "$OUT" --device {} > "$OUT/log_{}.txt" 2>&1; echo "   done {} ($(( $(date +%s) - '"$t0"' )) s)"'
wait
tail -n 1 "$OUT"/log_*.txt
grep -l "Traceback\|STOP" "$OUT"/log_*.txt
python "$S" summary --out "$OUT" > "$OUT/summary_log.txt" 2>&1
sed -n '6,16p' "$OUT/summary.md"
echo "H4 DIAG DONE after $(( $(date +%s) - t0 )) s"
