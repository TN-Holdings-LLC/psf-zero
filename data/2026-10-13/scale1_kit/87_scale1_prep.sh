#!/bin/bash
# Step 1 of SCALE1 (VPN ON; no IBM job, no QPU time): checks the kit and PSF-Zero's release file, then SCALE1's full
# rehearsal on FakeFez (noiseless matrix-product-state sampling). Commits nothing; runs no pip.
export TZ=Europe/Zurich
D=${SCALE1_DATE:-$(date +%F)}
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
PSF_SHA=0e97c2e222d0fb3885be4ad4f30cf995d7d9fed7feb45005b9451be1b0169d6b  # psf_compile.py, release 2026-10-11.4
stop() { echo "STOP: $1"; exit 1; }
echo "== 87 start $(date '+%F %T %Z')"
pgrep -f "scale1_real[.]py|pytes[t]" > /dev/null && stop "a run is already going"
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || stop "the kit's files differ from SHA256SUMS"
echo "kit files OK; SCALE1 prestatement: $(sha256sum "$S/SCALE1_preregistration.md" | cut -c1-64)"
cd "$R" || stop "repository not found at $R"
echo "repository head: $(git rev-parse --short HEAD); uncommitted changes to tracked files: $(git status --short --untracked-files=no | wc -l)"
[ "$(sha256sum psf_compile.py | cut -c1-64)" = "$PSF_SHA" ] || stop "psf_compile.py is not release 2026-10-11.4 (sha256 differs)"
source "$V" || stop "Python environment not found"
python -c "import qiskit, qiskit_aer, qiskit_ibm_runtime, psf_zero_core; print('qiskit', qiskit.__version__, '| aer', qiskit_aer.__version__, '| runtime', qiskit_ibm_runtime.__version__, '| core', getattr(psf_zero_core, 'CORE_VERSION', '?'))" || stop "a package is missing (nothing is installed by this kit)"
mkdir -p data/$D
T=$(date +%H%M%S)
echo "== SCALE1 rehearsal on FakeFez (about 5 minutes)"
timeout 3600 python -u "$S/scale1_real.py" rehearse --out "data/$D/scale1_rehearsal_$T" 2>&1 | tee "data/$D/scale1_rehearsal_$T.txt" | grep -E "compiling|NOT SUBMITTED|circuits in the main|manifest|REHEARSAL|Error|Traceback"
grep -q "REHEARSAL DONE" "data/$D/scale1_rehearsal_$T.txt" || stop "the rehearsal did not finish"
echo "== 87 DONE $(date '+%F %T %Z'). Next: turn the VPN OFF, then: bash $S/88_scale1_run.sh"
