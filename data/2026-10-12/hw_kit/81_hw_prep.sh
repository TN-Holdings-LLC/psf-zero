#!/bin/bash
# Step 1 of the real-device session (VPN ON; no IBM job, no QPU time): checks the kit, places prototypes c45 and c47
# in the repository's patches/, sets Stage 2's attempt-1 manifest aside as Addendum 211's Amendment 2 says, runs
# Stage 2's local checks (H0-H2) and HW1's full rehearsal on FakeKingston. Commits nothing; runs no pip.
export TZ=Europe/Zurich
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
echo "== 81 start $(date '+%F %T %Z')"
pgrep -f "hw1_real.py|xor_stage2_real.py" > /dev/null && stop "a run is already going"
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || stop "the kit's files differ from SHA256SUMS"
echo "kit files OK; HW1 prestatement: $(sha256sum "$S/HW1_preregistration.md" | cut -c1-64)"
cd "$R" || stop "repository not found at $R"
echo "repository head: $(git rev-parse --short HEAD) (expected 2222d55)"
source "$V" || stop "Python environment not found"
python -c "import qiskit, qiskit_aer, qiskit_ibm_runtime, pennylane, psf_zero_core; print('qiskit', qiskit.__version__, '| aer', qiskit_aer.__version__, '| runtime', qiskit_ibm_runtime.__version__, '| pennylane', pennylane.__version__, '| core', getattr(psf_zero_core, 'CORE_VERSION', '?'))" || stop "a package is missing (nothing is installed by this kit)"
for d in psf_compile_c45_2026-10-12 psf_compile_c47_2026-10-12; do
  mkdir -p "patches/$d"
  for f in "$S/patches/$d"/*; do
    b=$(basename "$f")
    if [ -e "patches/$d/$b" ]; then cmp -s "$f" "patches/$d/$b" || stop "patches/$d/$b exists and differs"
    else cp "$f" "patches/$d/$b"; fi
  done
done
echo "prototypes in patches/: c45 $(sha256sum patches/psf_compile_c45_2026-10-12/psf_compile.py | cut -c1-16), c47 $(sha256sum patches/psf_compile_c47_2026-10-12/psf_compile.py | cut -c1-16)"
M=stage2_manifest_2026-09-28.json; A=stage2_manifest_2026-09-28_attempt1_canceled.json
if [ -e "$M" ]; then
  [ -e "$A" ] && stop "both $M and $A exist: send this message"
  mv "$M" "$A" && echo "attempt 1's manifest set aside as $A (Addendum 211, Amendment 2)"
fi
ls -la stage2_* 2>/dev/null
mkdir -p data/2026-10-12/stage2
T=$(date +%H%M%S)
echo "== Stage 2 local checks (the IBM part at the end may fail with the VPN on; that is expected)"
timeout 1800 python -u benchmarks/xor_stage2_real.py preflight 2>&1 | tee "data/2026-10-12/stage2/stage2_preflight_vpn_on_$T.txt" | grep -E "SCRIPT|LOADED|H0|H1|H2|PREFLIGHT|account|Real devices"
for h in H0 H1 H2; do grep -q "^$h .*PASS" "data/2026-10-12/stage2/stage2_preflight_vpn_on_$T.txt" || stop "Stage 2 $h did not pass"; done
echo "== HW1 rehearsal on FakeKingston (about 5 minutes)"
timeout 3600 python -u "$S/hw1_real.py" rehearse --out "data/2026-10-12/hw1_rehearsal_$T" 2>&1 | tee "data/2026-10-12/hw1_rehearsal_$T.txt" | grep -E "compiling|NOT SUBMITTED|circuits,|manifest|X1|REHEARSAL|Error|Traceback"
grep -q "REHEARSAL DONE" "data/2026-10-12/hw1_rehearsal_$T.txt" || stop "the rehearsal did not finish"
grep -q "NOT SUBMITTED" "data/2026-10-12/hw1_rehearsal_$T.txt" && echo "NOTE: some circuit failed a check in the rehearsal (see the log)"
echo "== 81 DONE $(date '+%F %T %Z'). Next: turn the VPN OFF, then: bash $S/82_hw_run.sh"
