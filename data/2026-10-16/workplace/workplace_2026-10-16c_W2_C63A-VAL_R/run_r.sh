#!/bin/sh
# C63A-VAL part R (workplace), restart-safe: one segment per test, each in its own out dir. A segment whose log ends
# with "C63-VAL DONE" is skipped; an unfinished one is moved aside (kept) and run again from its start.
# Run from the repository root (bundle q):  sh <dir>/run_r.sh <dir of w34>
W=$(cd "$(dirname "$0")" && pwd); W34=$1; OUT=$W/run_r; mkdir -p $OUT
TESTS="tfim_n12 tfim_n16 tfimP_n12 tfimP_n16 tfim_n20_r3 heis_n12_r2 tfim_n40_r2 hhI_n16 hhI_n20 ring12_tfim hhtree_heis_n16 ghz_n12 chain_cz_n12 chain_cz_n16 ring_cz_n12 hh_cz_n16 hh_cz_n20 cx_chain_n12 cx_chain_n16"
for t in $TESTS; do
  d=$OUT/R_$t
  if [ -f $d.log ] && grep -q "C63-VAL DONE" $d.log; then continue; fi
  if [ -e $d ]; then n=1; while [ -e $d.killed$n ]; do n=$((n+1)); done; mv $d $d.killed$n; mv $d.log $d.killed$n/ 2>/dev/null; fi
  OMP_NUM_THREADS=1 C63_FILE=$W34/w3/c63a/psf_compile.py C63_ONLY=$t python -u $W34/c63_val.py $d R > $d.log 2>&1
done
