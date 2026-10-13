#!/bin/bash
# Step 2 of HW3 (VPN OFF): prepare (device rule, compile on the device's Target now, manifest before submission),
# submit ONE job, fetch the results. Runs no git and no pip. Each step refuses to run twice (manifest on disk).
#   bash 96_hw3_run.sh          all three steps
#   bash 96_hw3_run.sh from 3   resume at step 3 (e.g. after the wait limit)
export TZ=Europe/Zurich
D=${HW3_DATE:-$(date +%F)}
git() { echo "STOP: git is not used in this step"; exit 1; }
pip() { echo "STOP: pip is not used in this step"; exit 1; }
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP at step $STEP: $1"; exit 1; }
FROM=1; [ "$1" = from ] && FROM=$2
cd "$R" || { echo "STOP: repository not found"; exit 1; }
source "$V" || { echo "STOP: Python environment not found"; exit 1; }
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || { echo "STOP: the kit's files differ from SHA256SUMS"; exit 1; }
cmp -s "$S/patches/psf_compile_c48_2026-10-12/psf_compile.py" patches/psf_compile_c48_2026-10-12/psf_compile.py || { echo "STOP: run 95 first (c48 not in patches/)"; exit 1; }
OUT=data/$D/hw3; LG=data/$D/hw3_logs
mkdir -p "$LG"
echo "== 96 start $(date '+%F %T %Z'); HW3 prestatement $(sha256sum "$S/HW3_preregistration.md" | cut -c1-16)"
STEP=1; if [ "$FROM" -le 1 ]; then
  echo "== 1/3 prepare (device rule, compile on the device's Target now, checks, manifest before submission; a few minutes)"
  timeout 3600 python -u "$S/hw3_real.py" prepare --out "$OUT" 2>&1 | tee "$LG/hw3_prepare.txt" | grep -E "device rule|compiling|NOT SUBMITTED|not submitted|circuits,|manifest|STOP|Error|Traceback|refuses|not prepared|Neither"
  grep -q "manifest written before any submission" "$LG/hw3_prepare.txt" || stop "not prepared (nothing submitted)"
fi
STEP=2; if [ "$FROM" -le 2 ]; then
  echo "== 2/3 submit (one job)"
  timeout 900 python -u "$S/hw3_real.py" submit --out "$OUT" 2>&1 | tee "$LG/hw3_submit.txt" | tail -3
  grep -q "submitted job" "$LG/hw3_submit.txt" || stop "not submitted"
fi
STEP=3; if [ "$FROM" -le 3 ]; then
  echo "== 3/3 results (waits up to 40 minutes)"
  timeout 2700 python -u "$S/hw3_real.py" fetch --out "$OUT" --wait-min 40 2>&1 | tee "$LG/hw3_fetch.txt" | tail -4
  grep -q "saved hw3_raw_counts.json" "$LG/hw3_fetch.txt" || stop "results not saved yet: run 'bash $S/96_hw3_run.sh from 3' later"
fi
mkdir -p "$S/results"
cp -r "$OUT" "$LG" "$S/results/"
echo "== 96 DONE $(date '+%F %T %Z'). Results copied to the kit's results/. Turn the VPN back ON, then: bash $S/97_hw3_record.sh"
