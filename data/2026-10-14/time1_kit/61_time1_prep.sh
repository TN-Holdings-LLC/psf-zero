#!/bin/bash
# Step 1 of TIME1 (VPN ON; no IBM job, no QPU time): checks the kit and the repository (prototype c49 must be in
# patches/ with the hash fixed here), runs c49's tests and TIME1's full rehearsal on FakeFez with a noiseless local
# simulator. Commits nothing; runs no pip.
export TZ=Europe/Zurich
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
echo "== 61 start $(date '+%F %T %Z')"
pgrep -f "time1_real[.]py|pytes[t]" > /dev/null && stop "a run is already going"
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || stop "the kit's files differ from SHA256SUMS"
echo "kit files OK; TIME1 prestatement: $(sha256sum "$S/TIME1_preregistration.md" | cut -c1-64)"
cd "$R" || stop "repository not found at $R"
echo "repository head: $(git rev-parse --short HEAD); uncommitted changes to tracked files: $(git status --short --untracked-files=no | wc -l)"
source "$V" || stop "Python environment not found"
python -c "import qiskit, qiskit_aer, qiskit_ibm_runtime, psf_zero_core; print('qiskit', qiskit.__version__, '| aer', qiskit_aer.__version__, '| runtime', qiskit_ibm_runtime.__version__, '| core', getattr(psf_zero_core, 'CORE_VERSION', '?'))" || stop "a package is missing (nothing is installed by this kit)"
C=$(sha256sum patches/psf_compile_c49_2026-10-14/psf_compile.py 2>/dev/null | cut -c1-64)
[ "$C" = 5544d7c9f10d615c041eaf273af311573151a2548690b2192d866509bdaaa8d0 ] || stop "c49 in patches/ is missing or differs (merge the bundle first)"
echo "c49 OK (5544d7c9...)"
mkdir -p data/2026-10-14
T=$(date +%H%M%S)
echo "== c49's tests (about two minutes)"
timeout 2400 python -m pytest -q -p no:warnings patches/psf_compile_c49_2026-10-14/test_c49.py patches/psf_compile_c49_2026-10-14/test_c48_on_c49.py 2>&1 | tee "data/2026-10-14/time1_c49_tests_$T.txt" | tail -2
grep -qE "^[0-9]+ passed" "data/2026-10-14/time1_c49_tests_$T.txt" && ! grep -q "failed" "data/2026-10-14/time1_c49_tests_$T.txt" || stop "c49's tests did not all pass (see data/2026-10-14/time1_c49_tests_$T.txt)"
echo "== TIME1 rehearsal on FakeFez (about 10 minutes)"
timeout 3600 python -u "$S/time1_real.py" rehearse --out "data/2026-10-14/time1_rehearsal_$T" 2>&1 | tee "data/2026-10-14/time1_rehearsal_$T.txt" | grep -E "compiling|NOT SUBMITTED|not submitted|circuits,|manifest|REHEARSAL|Error|Traceback|STOP"
grep -q "REHEARSAL DONE" "data/2026-10-14/time1_rehearsal_$T.txt" || stop "the rehearsal did not finish"
grep -q "NOT SUBMITTED" "data/2026-10-14/time1_rehearsal_$T.txt" && echo "NOTE: some circuit failed a check in the rehearsal (see the log)"
echo "== 61 DONE $(date '+%F %T %Z'). Next: turn the VPN OFF, then: bash $S/62_time1_run.sh"
