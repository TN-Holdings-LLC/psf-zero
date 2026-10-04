#!/usr/bin/env bash
# run_suite_2026-10-04.sh -- the unattended home suite of 2026-10-04 (Addendum 334), in this order:
#   1. STALE  (pre-registered): smoke run (plumbing only), scored run (216 jobs), score
#   2. HYBRID (exploratory diagnosis, not a test): smoke run on one device, the nine devices, summary
#   3. WIDE   (pre-registered): smoke run (plumbing and timing), scored run (216 jobs), score
# A part whose smoke run fails (a Traceback or STOP in a log, or a missing output file) is skipped: its scored run is
# not started, and the next part still runs. Nothing is committed by this script.
#   bash benchmarks/run_suite_2026-10-04.sh <repo> <out root>
#   PAR=<n> parallel jobs (default 6); it changes only the wall time.
# Progress: tail -n 5 <out root>/suite_log.txt        Finished: the last line is "SUITE DONE"
set -uo pipefail
REPO="$(cd "$1" && pwd)"; ROOT="$2"; mkdir -p "$ROOT"; ROOT="$(cd "$ROOT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
HYB="$REPO/data/2026-10-04/hybrid/diag/hybrid_diag.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=FALSE
export PAR="${PAR:-6}" REPO
LOG="$ROOT/suite_log.txt"
DEVS="FakeKingston FakeFez FakeMarrakesh FakeAachen FakeTorino FakeAuckland FakeHanoiV2 FakeAlgiers FakeGeneva"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if ! grep -q '^VERSION = "2026-10-03.3"' "$REPO/psf_compile.py"; then
  say "STOP: $REPO/psf_compile.py is not release 2026-10-03.3"; exit 1
fi
{
  echo "date_utc $(date -u +%FT%TZ)"
  lscpu | grep -E "Model name|^CPU\(s\)"
  python -c "import sys, qiskit, qiskit_aer; print('python', sys.version.split()[0], 'qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)"
  sha256sum "$HERE/stale_eval.py" "$HERE/wide_eval.py" "$HYB" "$0"
  git -C "$REPO" log --oneline -1
  git -C "$REPO" status --short | head -20
  echo "PAR $PAR"
} > "$ROOT/env.txt" 2>&1
cat "$ROOT/env.txt" | tee -a "$LOG"

run_one() {   # <script> <out> <smoke flag or ""> <device> <arm> <family>
  local t=$(date +%s)
  python -u "$1" run --repo "$REPO" --out "$2" --device "$4" --arm "$5" --family "$6" $3 > "$2/log_$4_$5_$6.txt" 2>&1
  echo "   done $4 $5 $6 ($(( $(date +%s) - t )) s)"
}
export -f run_one

jobs216() {   # <script> <out> <smoke flag or "">
  for f in F3 F4 F1 F2 F6 F5; do for a in A7 R3 R2 L3T; do for d in $DEVS; do
    echo "$1 $2 ${3:-_} $d $a $f"; done; done; done |
    xargs -P "$PAR" -L 1 bash -c 'run_one "$1" "$2" "$( [ "$3" = _ ] || echo "$3")" "$4" "$5" "$6"' _ >> "$LOG"
}

broken() {    # <out dir> <prefix> <expected json count>: prints a reason if the run is broken
  local n; n=$(ls "$1"/"$2"_*.json 2>/dev/null | wc -l)
  if grep -l "Traceback\|STOP" "$1"/log_*.txt > /dev/null 2>&1; then
    echo "errors in $(grep -l "Traceback\|STOP" "$1"/log_*.txt | head -3 | xargs -n1 basename | tr '\n' ' ')"
  elif [ "$n" -ne "$3" ]; then echo "$n of $3 output files"; fi
}

part() {      # <name> <script> <prefix>
  local name="$1" S="$2" P="$3" t0
  say "== $name smoke run (plumbing only)"
  mkdir -p "$ROOT/${P}_smoke"; t0=$(date +%s)
  jobs216 "$S" "$ROOT/${P}_smoke" --smoke
  local why; why=$(broken "$ROOT/${P}_smoke" "$P" 216)
  if [ -n "$why" ]; then
    say "!! $name SMOKE FAILED ($why); its scored run is skipped"
    grep -h -A3 "Traceback\|STOP" "$ROOT/${P}_smoke"/log_*.txt | head -20 | tee -a "$LOG"
    return 1
  fi
  python "$S" score --out "$ROOT/${P}_smoke" > "$ROOT/${P}_smoke/score_log.txt" 2>&1
  say "   $name smoke OK after $(( $(date +%s) - t0 )) s"
  say "== $name scored run"
  mkdir -p "$ROOT/$P"; t0=$(date +%s)
  jobs216 "$S" "$ROOT/$P" ""
  say "   $name jobs finished after $(( $(date +%s) - t0 )) s; $(broken "$ROOT/$P" "$P" 216)"
  grep -h "Traceback\|Error" "$ROOT/$P"/log_*.txt | head -5 | tee -a "$LOG"
  python "$S" score --out "$ROOT/$P" > "$ROOT/$P/score_log.txt" 2>&1
  grep -E "^P0|^- H[0-9]" "$ROOT/$P/score.md" | tee -a "$LOG"
  say "== $name DONE"
}

hybrid() {
  say "== HYBRID smoke run (one device, 2 circuits per family)"
  local O="$ROOT/hybrid_smoke" t0; mkdir -p "$O"
  python -u "$HYB" run --repo "$REPO" --out "$O" --device FakeGeneva --smoke > "$O/log_FakeGeneva.txt" 2>&1
  if grep -q "Traceback\|STOP" "$O/log_FakeGeneva.txt" || [ ! -f "$O/hybrid_FakeGeneva_smoke.json" ]; then
    say "!! HYBRID SMOKE FAILED; the diagnosis is skipped"; tail -n 15 "$O/log_FakeGeneva.txt" | tee -a "$LOG"
    return 1
  fi
  say "== HYBRID diagnosis, nine devices"
  O="$ROOT/hybrid"; mkdir -p "$O"; t0=$(date +%s)
  export HYB O
  for d in $DEVS; do echo "$d"; done |
    xargs -P "$PAR" -I{} bash -c 'python -u "$HYB" run --repo "$REPO" --out "$O" --device {} > "$O/log_{}.txt" 2>&1; echo "   done hybrid {}"' >> "$LOG"
  say "   HYBRID finished after $(( $(date +%s) - t0 )) s"
  grep -h "Traceback\|STOP" "$O"/log_*.txt | head -5 | tee -a "$LOG"
  python "$HYB" summary --out "$O" > "$O/summary_log.txt" 2>&1
  sed -n '8,18p' "$O/summary.md" | tee -a "$LOG"
  say "== HYBRID DONE"
}

T0=$(date +%s)
part STALE "$HERE/stale_eval.py" stale
hybrid
part WIDE "$HERE/wide_eval.py" wide
say "SUITE DONE after $(( ($(date +%s) - T0) / 60 )) min"
