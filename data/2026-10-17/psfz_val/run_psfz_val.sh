#!/bin/sh
# PSFZ-VAL (the authors' cloud): psfz built from its committed sources, then two processes of devices. From the
# PSF-Zero root:  sh data/2026-10-17/psfz_val/run_psfz_val.sh <out dir>
OUT=$1; [ -e "$OUT" ] && { echo "STOP: $OUT exists"; exit 1; }; mkdir -p $OUT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=${PY:-python}
V=data/2026-10-17/psfz_val
echo "start $(TZ=Europe/Zurich date '+%F %T %Z')" > $OUT/times.txt
( cd rust/psfz && cargo build --release --offline -q ) || { echo "STOP: psfz does not build" >> $OUT/times.txt; exit 1; }
sha256sum rust/psfz/Cargo.toml rust/psfz/src/*.rs rust/psfz/target/release/psfz psf_compile.py $V/*.py $V/*.md >> $OUT/times.txt
$PY -c "import qiskit, qiskit_aer, qiskit_ibm_runtime as r; print('qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__, 'runtime', r.__version__)" >> $OUT/times.txt
cargo --version >> $OUT/times.txt
( $PY -u $V/psfz_val.py $OUT/rows_a.jsonl FakeAachen FakeBoston FakeKingston FakePittsburgh FakeFez FakeMarrakesh FakeTorino > $OUT/a.log 2>&1; echo "process a end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt ) &
( $PY -u $V/psfz_val.py $OUT/rows_b.jsonl FakeKyiv FakeOsaka FakeKawasaki FakeQuebec FakeSherbrooke FakeBrisbane > $OUT/b.log 2>&1; echo "process b end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt ) &
wait
echo "end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt
$PY $V/score_psfz_val.py $OUT/rows_a.jsonl $OUT/rows_b.jsonl > $OUT/score.txt 2>&1
