# C25-ID2

git head 759f220, uncommitted tracked changes: none; 152 tests (expected 152), 558 jobs, 4 at a time, 14 CPUs, Python 3.11.9

| ID | prediction | value | verdict |
|---|---|---|---|
| J0 | the run is as locked | 152 tests, every arm on every job, versions and virtual clock in every record: True | **PASS** |
| J1 | REL2 = REL (control) | 177 identical of 185 | **REFUTED** |
| J2 | C25 = REL | 177 identical of 185 | **REFUTED** |
| J3 | the same failures in every arm | REL 1, REL2 1, C25 1 | **CONFIRMED** |
  - REL2 differs: test_QASMBench_large[bv_n140-linear] (default)
  - REL2 differs: test_QASMBench_large[bv_n30-square] (default)
  - REL2 differs: test_QASMBench_small[inverseqft_n4-linear] (default)
  - REL2 differs: test_QASMBench_small[qec_sm_n5-all-to-all] (default)
  - REL2 differs: test_circSU2_100_transpile (default)
  - REL2 differs: test_circSU2_100_transpile (recommended)
  - REL2 differs: test_circSU2_89_transpile (default)
  - REL2 differs: test_circSU2_89_transpile (recommended)
  - C25 differs: test_QASMBench_large[bv_n140-linear] (default)
  - C25 differs: test_QASMBench_large[bv_n30-square] (default)
  - C25 differs: test_QASMBench_small[inverseqft_n4-linear] (default)
  - C25 differs: test_QASMBench_small[qec_sm_n5-all-to-all] (default)
  - C25 differs: test_circSU2_100_transpile (default)
  - C25 differs: test_circSU2_100_transpile (recommended)
  - C25 differs: test_circSU2_89_transpile (default)
  - C25 differs: test_circSU2_89_transpile (recommended)
  - failed (REL): test_feynman_transpile[hwb10.qasm] (recommended): timeout 3600 s
  - failed (REL2): test_feynman_transpile[hwb10.qasm] (recommended): timeout 3600 s
  - failed (C25): test_feynman_transpile[hwb10.qasm] (recommended): timeout 3600 s

SUMMARY {"J0": "PASS", "J1": "REFUTED", "J2": "REFUTED", "J3": "CONFIRMED"}
