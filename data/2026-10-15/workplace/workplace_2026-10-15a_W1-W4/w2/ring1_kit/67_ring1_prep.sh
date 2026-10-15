#!/bin/bash
# Step 1 of RING1 (VPN ON; no IBM job, no QPU time): checks the kit and the repository (release 2026-10-14.1:
# psf_compile.py at the root and benchmarks/psf_ai_compile.py = a18, with the hashes fixed here), runs c57's and
# a18's tests (a18's on the release's psf_compile.py), then RING1's full rehearsal on FakeFez with a noiseless local
# simulator. With "noisy" it adds a rehearsal with FakeFez's noise model (reported only; about 5 minutes more).
# Commits nothing; runs no pip.
#   bash 67_ring1_prep.sh          tests + noiseless rehearsal
#   bash 67_ring1_prep.sh noisy    ... + noisy rehearsal
export TZ=Europe/Zurich
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
echo "== 67 start $(date '+%F %T %Z')"
pgrep -f "ring1_real[.]py|pytes[t]" > /dev/null && stop "a run is already going"
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || stop "the kit's files differ from SHA256SUMS"
echo "kit files OK; RING1 prestatement: $(sha256sum "$S/RING1_preregistration.md" | cut -c1-64)"
cd "$R" || stop "repository not found at $R"
echo "repository head: $(git rev-parse --short HEAD); uncommitted changes to tracked files: $(git status --short --untracked-files=no | wc -l)"
source "$V" || stop "Python environment not found"
python -c "import qiskit, qiskit_aer, qiskit_ibm_runtime, psf_zero_core; print('qiskit', qiskit.__version__, '| aer', qiskit_aer.__version__, '| runtime', qiskit_ibm_runtime.__version__, '| core', getattr(psf_zero_core, 'CORE_VERSION', '?'))" || stop "a package is missing (nothing is installed by this kit)"
chk() { [ "$(sha256sum "$1" 2>/dev/null | cut -c1-64)" = "$2" ] || stop "$1 is missing or differs (merge bundle n / release 2026-10-14.1 first)"; echo "$1 OK (${2:0:8}...)"; }
chk psf_compile.py 23d4aa4d1cdea7729d323c82d63386f43ccb302fc28ffe50f9ae58363459d0f8
chk benchmarks/psf_ai_compile.py 94bd54eeed0d2c2941d3dda574737acf029a71041ee47f572710e24ce545e5c1
chk data/2026-10-14/time1_kit/time1_real.py 171f8922446443bf8a27e1552f1ff604a4d243346759bc8f1c1fdad7b1025505
mkdir -p data/2026-10-15
T=$(date +%H%M%S)
echo "== c57's and a18's tests (a18's with PSF_COMPILE = the release's psf_compile.py; about a minute)"
timeout 3600 python -m pytest -q -p no:warnings patches/psf_compile_c57_2026-10-14/test_c57.py 2>&1 | tee "data/2026-10-15/ring1_tests_$T.txt" | tail -2
PSF_COMPILE="$R/psf_compile.py" timeout 3600 python -m pytest -q -p no:warnings patches/psf_ai_compile_a18_2026-10-14/test_a18.py 2>&1 | tee -a "data/2026-10-15/ring1_tests_$T.txt" | tail -2
[ "$(grep -cE "^[0-9]+ passed" "data/2026-10-15/ring1_tests_$T.txt")" = 2 ] && ! grep -qE "[0-9]+ (failed|error)" "data/2026-10-15/ring1_tests_$T.txt" || stop "the tests did not all pass (see data/2026-10-15/ring1_tests_$T.txt)"
echo "== RING1 rehearsal on FakeFez, noiseless (about 2 minutes)"
timeout 3600 python -u "$S/ring1_real.py" rehearse --out "data/2026-10-15/ring1_rehearsal_$T" 2>&1 | tee "data/2026-10-15/ring1_rehearsal_$T.txt" | grep -E "compiling|NOT SUBMITTED|incomplete|circuits|manifest|REHEARSAL|Error|Traceback|STOP|stop"
grep -q "REHEARSAL DONE" "data/2026-10-15/ring1_rehearsal_$T.txt" || stop "the rehearsal did not finish"
grep -qE "NOT SUBMITTED|incomplete" "data/2026-10-15/ring1_rehearsal_$T.txt" && echo "NOTE: some circuit failed a check in the rehearsal (see the log)"
if [ "$1" = noisy ]; then
  echo "== RING1 rehearsal on FakeFez with its noise model (reported only; about 5 minutes)"
  timeout 3600 python -u "$S/ring1_real.py" rehearse --noisy --out "data/2026-10-15/ring1_rehearsal_noisy_$T" 2>&1 | tee "data/2026-10-15/ring1_rehearsal_noisy_$T.txt" | grep -E "^\| [46] |RA |RB |REHEARSAL|Error|Traceback"
fi
echo "== 67 DONE $(date '+%F %T %Z'). Next: turn the VPN OFF, then: bash $S/68_ring1_run.sh"
