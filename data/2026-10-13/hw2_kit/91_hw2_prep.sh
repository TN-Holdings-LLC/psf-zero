#!/bin/bash
# Step 1 of HW2 (VPN ON; no IBM job, no QPU time): checks the kit, places prototypes c44, c45, c47 and c48 in the
# repository's patches/ (a file already there must be identical), runs c48's tests, and HW2's full rehearsal on
# FakeFez with a noiseless local simulator. Commits nothing; runs no pip.
export TZ=Europe/Zurich
S=$(dirname "$(readlink -f "$0")")
R=${PSF_REPO:-~/psf_zero_fresh_test}
V=${PSF_VENV:-~/psf_zero_fresh_test/psf_zero_wsl_env_fresh/bin/activate}
stop() { echo "STOP: $1"; exit 1; }
echo "== 91 start $(date '+%F %T %Z')"
pgrep -f "hw2_real[.]py|pytes[t]" > /dev/null && stop "a run is already going"
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || stop "the kit's files differ from SHA256SUMS"
echo "kit files OK; HW2 prestatement: $(sha256sum "$S/HW2_preregistration.md" | cut -c1-64)"
cd "$R" || stop "repository not found at $R"
echo "repository head: $(git rev-parse --short HEAD); uncommitted changes to tracked files: $(git status --short --untracked-files=no | wc -l)"
source "$V" || stop "Python environment not found"
python -c "import qiskit, qiskit_aer, qiskit_ibm_runtime, psf_zero_core; print('qiskit', qiskit.__version__, '| aer', qiskit_aer.__version__, '| runtime', qiskit_ibm_runtime.__version__, '| core', getattr(psf_zero_core, 'CORE_VERSION', '?'))" || stop "a package is missing (nothing is installed by this kit)"
for d in psf_compile_c44_2026-10-11 psf_compile_c45_2026-10-12 psf_compile_c47_2026-10-12 psf_compile_c48_2026-10-12; do
  mkdir -p "patches/$d"
  for f in "$S/patches/$d"/*; do
    b=$(basename "$f")
    if [ -e "patches/$d/$b" ]; then cmp -s "$f" "patches/$d/$b" || stop "patches/$d/$b exists and differs"
    else cp "$f" "patches/$d/$b"; fi
  done
done
echo "prototypes in patches/: c45 $(sha256sum patches/psf_compile_c45_2026-10-12/psf_compile.py | cut -c1-16) (expected dc51b0492f01b4c4), c48 $(sha256sum patches/psf_compile_c48_2026-10-12/psf_compile.py | cut -c1-16) (expected 0400100f416421db)"
mkdir -p data/2026-10-13
T=$(date +%H%M%S)
echo "== c48's tests (about a minute)"
timeout 1800 python -m pytest -q -p no:warnings patches/psf_compile_c48_2026-10-12/test_c48.py 2>&1 | tee "data/2026-10-13/hw2_c48_tests_$T.txt" | tail -2
grep -qE "^[0-9]+ passed" "data/2026-10-13/hw2_c48_tests_$T.txt" && ! grep -q "failed" "data/2026-10-13/hw2_c48_tests_$T.txt" || stop "c48's tests did not all pass (see data/2026-10-13/hw2_c48_tests_$T.txt)"
echo "== HW2 rehearsal on FakeFez (about 5 minutes)"
timeout 3600 python -u "$S/hw2_real.py" rehearse --out "data/2026-10-13/hw2_rehearsal_$T" 2>&1 | tee "data/2026-10-13/hw2_rehearsal_$T.txt" | grep -E "compiling|NOT SUBMITTED|not submitted|circuits,|manifest|Replication|REHEARSAL|Error|Traceback"
grep -q "REHEARSAL DONE" "data/2026-10-13/hw2_rehearsal_$T.txt" || stop "the rehearsal did not finish"
grep -q "NOT SUBMITTED" "data/2026-10-13/hw2_rehearsal_$T.txt" && echo "NOTE: some circuit failed a check in the rehearsal (see the log)"
echo "== 91 DONE $(date '+%F %T %Z'). Next: turn the VPN OFF, then: bash $S/92_hw2_run.sh"
