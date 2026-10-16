#!/bin/sh
# Restart-safe runner for D1 / D5 (C64-PREREG.md): every segment has its own out dir; a segment whose log ends with
# "C64 CHECK DONE" is skipped; an unfinished one is moved aside (kept, reported) and run again from its start.
# Run from the repository root (bundle q). Segments: X (refusals); R, one per test; D, one per device.
W=$(cd "$(dirname "$0")" && pwd); OUT=$W/run_d1; mkdir -p $OUT  # (the run used the workplace folder of these files)
TESTS="tfim_n12 tfim_n16 tfimP_n12 tfimP_n16 tfim_n20_r3 heis_n12_r2 tfim_n40_r2 hhI_n16 hhI_n20 ring12_tfim hhtree_heis_n16 ghz_n12 chain_cz_n12 chain_cz_n16 ring_cz_n12 hh_cz_n16 hh_cz_n20 cx_chain_n12 cx_chain_n16"
seg() {  # name, parts, env
  d=$OUT/$1
  if [ -f $d.log ] && grep -q "C64 CHECK DONE" $d.log; then return; fi
  if [ -e $d ]; then n=1; while [ -e $d.killed$n ]; do n=$((n+1)); done; mv $d $d.killed$n; mv $d.log $d.killed$n/ 2>/dev/null; fi
  env $3 python -u $W/c64_check.py $d $2 > $d.log 2>&1
}
seg X X ""
for t in $TESTS; do seg R_$t R "C64_ONLY=$t"; done
for dv in FakeMarrakesh FakeKingston FakeTorino snapshot_marrakesh; do seg D_$dv D "C64_DEVICES=$dv"; done
TZ=Europe/Zurich date "+D1 segments all done %F %H:%M:%S %Z" >> $W/results/times.txt
