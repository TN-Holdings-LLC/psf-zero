# C29-ID

git head a00ade5, uncommitted tracked changes: none; 152 tests (expected 152), 186 test-calls, 558 jobs, 4 at a time, 12 CPUs, Python 3.12.13

| ID | prediction | value | verdict |
|---|---|---|---|
| U0 | the run is as locked | 152 tests, every arm on every test-call, versions and virtual clock in every record, no uncommitted change: True | **PASS** |
| U1 | C29 = REL by value wherever REL2 = REL | 183 identical of 183 scored | **CONFIRMED** |
| U2 | C29 fails nowhere that REL and REL2 do not | C29 0 failures, 0 new | **CONFIRMED** |
| U3 | hwb10, recommended: C29's wall time under 0.2 of REL's | C29 82 s | **REFUTED** |

Reported without prediction:

- control: REL2 = REL by value on 183 of 185 test-calls both finished; by sig_hash on 177
- not scored (3): REL2 differs from REL, or an arm failed
- item 56 skipped something (C29's FEASIBILITY_STATS) on 1 test-calls
- recommended calls finished in all three arms (33): summed compile time REL 2848 s, REL2 2840 s, C29 2795 s
  - REL2 differs (not scored): test_QASMBench_large[bv_n140-linear] (default)
  - REL2 differs (not scored): test_QASMBench_large[bv_n30-square] (default)
  - C29 skipped: test_feynman_transpile[hwb10.qasm] (recommended): {'candidate': 2, 'resynthesis': 2}
  - failed (REL): test_feynman_transpile[hwb10.qasm] (recommended): timeout 3600 s
  - failed (REL2): test_feynman_transpile[hwb10.qasm] (recommended): timeout 3600 s

SUMMARY {"U0": "PASS", "U1": "CONFIRMED", "U2": "CONFIRMED", "U3": "REFUTED"}
