#!/bin/sh
# C66-VAL (home, the authors' cloud): C63-VAL's parts R and D on prototype c66, the default call against release
# 2026-10-16.1 on every C63-VAL input, and C61-VAL part K's walk. Two processes. Run from the PSF-Zero root:
#   sh data/2026-10-16/c66_val/run_c66_val.sh <out dir>
OUT=$1; [ -e "$OUT" ] && { echo "STOP: $OUT exists"; exit 1; }; mkdir -p $OUT
W34=data/2026-10-15/workplace/workplace_2026-10-15c_W1-W3/w34
C66=patches/psf_compile_c66_2026-10-16/psf_compile.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=${PY:-python}
echo "start $(TZ=Europe/Zurich date '+%F %T %Z')" > $OUT/times.txt
sha256sum $C66 data/2026-10-16/c66_val/*.py data/2026-10-16/c66_val/*.md >> $OUT/times.txt
( for t in tfim_n12 tfim_n16 tfimP_n12 tfimP_n16 tfim_n20_r3 heis_n12_r2 tfim_n40_r2 hhI_n16 hhI_n20 ring12_tfim; do
    C63_FILE=$C66 C63_ONLY=$t $PY -u $W34/c63_val.py $OUT/R/R_$t R > $OUT/R_$t.log 2>&1; done
  C63_FILE=$C66 $PY -u $W34/c63_val.py $OUT/D D > $OUT/D.log 2>&1
  echo "process 1 end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt ) &
( for t in hhtree_heis_n16 ghz_n12 chain_cz_n12 chain_cz_n16 ring_cz_n12 hh_cz_n16 hh_cz_n20 cx_chain_n12 cx_chain_n16; do
    C63_FILE=$C66 C63_ONLY=$t $PY -u $W34/c63_val.py $OUT/R/R_$t R > $OUT/R_$t.log 2>&1; done
  W3_A_FILE=$C66 $PY -u data/2026-10-16/c66_val/walk_c66.py A $OUT/walk.json > $OUT/walk.log 2>&1
  $PY -u data/2026-10-16/c66_val/default_call.py $OUT/default_call.json > $OUT/default_call.log 2>&1
  echo "process 2 end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt ) &
wait
echo "end $(TZ=Europe/Zurich date '+%F %T %Z')" >> $OUT/times.txt
