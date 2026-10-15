#!/bin/bash
# TOQB loop pilot, n = 6, L = 8: Qiskit level 2's template against PSF-Zero's (a22 + c61), two SPSA seeds in parallel.
# Home, WSL; no network needed. Uses the clones pushed on 2026-10-15: TOQB e7527cc, psf-zero 7116bc2.
# Writes the results under TOQB's results/ and a zip of them next to this script; never overwrites an earlier run.
set -u
TOQB=~/toqb_push_2026-10-15
PSF=~/psf_zero_fresh_test
DAY=${DAY:?set DAY to the day folder on the home PC}
OUT=results/loop-pilot-n6-2026-10-15
cd "$TOQB" || { echo "STOP: no $TOQB"; exit 1; }
git log --oneline -1 | grep -q '^e7527cc' || { echo "STOP: TOQB is not at e7527cc"; exit 1; }
git -C "$PSF" log --oneline -1 | grep -q '^7116bc2' || { echo "STOP: psf-zero is not at 7116bc2"; exit 1; }
[ -e "$OUT" ] && { echo "STOP: $OUT exists; a run never overwrites another"; exit 1; }
python -c "import qiskit, qiskit_aer, qiskit_ibm_runtime, numpy; print('qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)" \
  || { echo "STOP: needs qiskit, qiskit-aer and qiskit-ibm-runtime in this environment"; exit 1; }
cp "$DAY/plan_loop_pilot_n6_s11.json" "$DAY/plan_loop_pilot_n6_s12.json" .
echo "start $(TZ=Europe/Zurich date '+%Y-%m-%d %H:%M:%S %Z')"
for s in 11 12; do
  PSF_ZERO_REPO="$PSF" PSF_COMPILE_FILE="$PSF/patches/psf_compile_c61_2026-10-15/psf_compile.py" PYTHONWARNINGS=ignore \
    python -m toqb.task_loop run "plan_loop_pilot_n6_s$s.json" "$OUT/seed$s" > "$DAY/pilot_n6_seed$s.log" 2>&1 &
done
wait
echo "end $(TZ=Europe/Zurich date '+%Y-%m-%d %H:%M:%S %Z')"
python -m zipfile -c "$DAY/pilot_n6_results.zip" "$OUT" plan_loop_pilot_n6_s11.json plan_loop_pilot_n6_s12.json \
  && echo "== DONE: $DAY/pilot_n6_results.zip"
