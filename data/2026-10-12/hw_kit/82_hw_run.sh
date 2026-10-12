#!/bin/bash
# Step 2 of the real-device session (VPN OFF): Stage 2 attempt 2 (Addendum 211, unchanged) and HW1 + PSF-XOR (one
# job), then the results. Runs no git and no pip. Each submission refuses to run twice (manifests on disk).
#   bash 82_hw_run.sh          all six steps
#   bash 82_hw_run.sh from 5   resume at step 5 (e.g. after a wait limit)
export TZ=Europe/Zurich
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
ST=data/2026-10-12/stage2; OUT=data/2026-10-12/hw1; LG=data/2026-10-12/hw1_logs
mkdir -p "$ST" "$LG"
echo "== 82 start $(date '+%F %T %Z'); HW1 prestatement $(sha256sum "$S/HW1_preregistration.md" | cut -c1-16)"

STEP=1; if [ "$FROM" -le 1 ]; then
  echo "== 1/6 Stage 2 preflight (saved account, device list)"
  timeout 1800 python -u benchmarks/xor_stage2_real.py preflight 2>&1 | tee "$ST/stage2_preflight.txt" | tail -12
  grep -q "PREFLIGHT OK" "$ST/stage2_preflight.txt" || stop "Stage 2 preflight not OK (nothing submitted)"
fi
STEP=2; if [ "$FROM" -le 2 ]; then
  echo "== 2/6 Stage 2 submit (one job, 8 circuits)"
  timeout 3600 python -u benchmarks/xor_stage2_real.py submit 2>&1 | tee "$ST/stage2_submit.txt" | tail -12
  grep -q "Submitted job" "$ST/stage2_submit.txt" || stop "Stage 2 was not submitted"
fi
STEP=3; if [ "$FROM" -le 3 ]; then
  echo "== 3/6 HW1 prepare (device rule, compile on the device's Target now, checks, manifest before submission)"
  timeout 3600 python -u "$S/hw1_real.py" prepare --out "$OUT" 2>&1 | tee "$LG/hw1_prepare.txt" | grep -E "device rule|compiling|NOT SUBMITTED|circuits,|manifest|STOP|Error|Traceback|refuses|not prepared"
  grep -q "manifest written before any submission" "$LG/hw1_prepare.txt" || stop "HW1 not prepared (nothing submitted)"
fi
STEP=4; if [ "$FROM" -le 4 ]; then
  echo "== 4/6 HW1 submit (one job)"
  timeout 900 python -u "$S/hw1_real.py" submit --out "$OUT" 2>&1 | tee "$LG/hw1_submit.txt" | tail -3
  grep -q "submitted job" "$LG/hw1_submit.txt" || stop "HW1 was not submitted"
fi
STEP=5; if [ "$FROM" -le 5 ]; then
  echo "== 5/6 HW1 results (waits up to 40 minutes)"
  timeout 2700 python -u "$S/hw1_real.py" fetch --out "$OUT" --wait-min 40 2>&1 | tee "$LG/hw1_fetch.txt" | tail -4
  grep -q "saved hw1_raw_counts.json" "$LG/hw1_fetch.txt" || stop "HW1 results not saved yet: run 'bash $S/82_hw_run.sh from 5' later"
fi
STEP=6; if [ "$FROM" -le 6 ]; then
  echo "== 6/6 Stage 2 score (tries every minute, up to 40 times)"
  for i in $(seq 1 40); do
    timeout 900 python -u benchmarks/xor_stage2_real.py score > "$ST/stage2_score.txt" 2>&1
    grep -q "^Wrote" "$ST/stage2_score.txt" && break
    tail -1 "$ST/stage2_score.txt"; sleep 60
  done
  grep -q "^Wrote" "$ST/stage2_score.txt" || stop "Stage 2 not finished: run 'bash $S/82_hw_run.sh from 6' later"
  grep -E "S[1-4] |P0 " "$ST/stage2_score.txt"
fi
mkdir -p "$S/results"
cp stage2_manifest_2026-09-28*.json "$ST/" 2>/dev/null
cp -r "$ST" "$OUT" "$LG" "$S/results/"
echo "== 82 DONE $(date '+%F %T %Z'). Results copied to the kit's results/. Turn the VPN back ON, then: bash $S/83_hw_record.sh"
