#!/bin/sh
# C67-VAL (the authors' cloud): two processes, two devices each. From the PSF-Zero root:
#   sh data/2026-10-17/c67_val/run_c67_val.sh <out dir>
OUT=$1; [ -e "$OUT" ] && { echo "STOP: $OUT exists"; exit 1; }; mkdir -p $OUT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=${PY:-python}
V=data/2026-10-17/c67_val
echo "start $(TZ=Europe/Zurich date '+%F %T %Z')" > $OUT/times.txt
sha256sum patches/psf_compile_c67_2026-10-17/psf_compile.py $V/*.py $V/*.md >> $OUT/times.txt
$PY -c "import qiskit, qiskit_aer, qiskit_ibm_runtime as r; print('qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__, 'runtime', r.__version__)" >> $OUT/times.txt
( $PY -u $V/c67_val.py $OUT/rows_a.jsonl FakeSherbrooke FakeBrisbane > $OUT/a.log 2>&1; echo "process a end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt ) &
( $PY -u $V/c67_val.py $OUT/rows_b.jsonl FakeKyiv FakeOsaka > $OUT/b.log 2>&1; echo "process b end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt ) &
wait
echo "end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt
$PY $V/score_c67_val.py $OUT/rows_a.jsonl $OUT/rows_b.jsonl > $OUT/score.txt 2>&1
