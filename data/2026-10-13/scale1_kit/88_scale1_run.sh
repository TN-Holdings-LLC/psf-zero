#!/bin/bash
# Step 2 of SCALE1 (VPN OFF): prepare (device rule, compile on the device's Target now, checks, manifest before
# submission), submit the main job and the part-F job (if any), fetch the results. Runs no git and no pip. Each step
# refuses to run twice (manifest on disk).
#   bash 88_scale1_run.sh          all three steps
#   bash 88_scale1_run.sh from 3   resume at step 3 (e.g. after the wait limit)
export TZ=Europe/Zurich
D=${SCALE1_DATE:-$(date +%F)}
git() { echo "STOP: git is not used in this step"; exit 1; }
pip() { echo "STOP: pip is not used in this step"; exit 1; }
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
PSF_SHA=0e97c2e222d0fb3885be4ad4f30cf995d7d9fed7feb45005b9451be1b0169d6b
stop() { echo "STOP at step $STEP: $1"; exit 1; }
FROM=1; [ "$1" = from ] && FROM=$2
cd "$R" || { echo "STOP: repository not found"; exit 1; }
source "$V" || { echo "STOP: Python environment not found"; exit 1; }
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || { echo "STOP: the kit's files differ from SHA256SUMS"; exit 1; }
[ "$(sha256sum psf_compile.py | cut -c1-64)" = "$PSF_SHA" ] || { echo "STOP: psf_compile.py is not release 2026-10-11.4"; exit 1; }
OUT=data/$D/scale1; LG=data/$D/scale1_logs
mkdir -p "$LG"
echo "== 88 start $(date '+%F %T %Z'); SCALE1 prestatement $(sha256sum "$S/SCALE1_preregistration.md" | cut -c1-16)"
STEP=1; if [ "$FROM" -le 1 ]; then
  echo "== 1/3 prepare (device rule, compile 89 circuits on the device's Target now, checks, manifest before submission; a few minutes)"
  timeout 3600 python -u "$S/scale1_real.py" prepare --out "$OUT" 2>&1 | tee "$LG/scale1_prepare.txt" | grep -E "device rule|compiling|NOT SUBMITTED|circuits in the main|manifest|STOP|Error|Traceback|not prepared|Neither"
  grep -q "manifest written before any submission" "$LG/scale1_prepare.txt" || stop "not prepared (nothing submitted)"
fi
STEP=2; if [ "$FROM" -le 2 ]; then
  echo "== 2/3 submit (the main job, then part F if any)"
  timeout 900 python -u "$S/scale1_real.py" submit --out "$OUT" 2>&1 | tee "$LG/scale1_submit.txt" | tail -4
  grep -q "submitted jobs" "$LG/scale1_submit.txt" || stop "not submitted"
fi
STEP=3; if [ "$FROM" -le 3 ]; then
  echo "== 3/3 results (waits up to 40 minutes per job)"
  timeout 5400 python -u "$S/scale1_real.py" fetch --out "$OUT" --wait-min 40 2>&1 | tee "$LG/scale1_fetch.txt" | tail -5
  grep -q "saved all scale1 results" "$LG/scale1_fetch.txt" || stop "results not saved yet: run 'bash $S/88_scale1_run.sh from 3' later"
fi
mkdir -p "$S/results"
cp -r "$OUT" "$LG" "$S/results/"
echo "== 88 DONE $(date '+%F %T %Z'). Results copied to the kit's results/. Turn the VPN back ON, then: bash $S/89_scale1_record.sh"
