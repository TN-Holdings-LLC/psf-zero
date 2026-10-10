# C30-ID

git head 9de4513, uncommitted tracked changes: none; 152 tests (expected 152), 186 test-calls, 558 jobs, 4 at a time, 12 CPUs, Python 3.12.13

| ID | prediction | value | verdict |
|---|---|---|---|
| RC0 | the run is as locked | 152 tests, every arm on every test-call, versions, virtual clock and psf_zero_core57 2026-10-10.c30 in every record, no uncommitted change: True | **PASS** |
| RC1 | C30 = REL by value wherever REL2 = REL | 184 identical of 184 scored | **CONFIRMED** |
| RC2 | C30 fails nowhere that REL and REL2 do not | C30 0 failures, 0 new | **CONFIRMED** |
| RC3 | recommended calls with estimates: geometric mean of C30 / REL compile time <= 0.85 | 0.656 on 14 test-calls | **CONFIRMED** |

Reported without prediction:

- control: REL2 = REL by value on 184 of 186 test-calls both finished; on RC3's set, geometric mean of REL2 / REL time 0.978
- not scored (2): REL2 differs from REL, or an arm failed
- RC3's set, summed compile time: REL 1641 s, REL2 1638 s, C30 141 s
- C30's estimate and check calls: 229 in psf_zero_core57, 0 in Python
  - REL2 differs (not scored): test_QASMBench_large[bv_n140-linear] (default)
  - REL2 differs (not scored): test_QASMBench_large[bv_n30-square] (default)
  - RC3: test_BVlike_simplification_transpile: REL 0.6 s, C30 0.6 s (0.95)
  - RC3: test_feynman_transpile[barenco_tof_5.qasm]: REL 0.4 s, C30 0.3 s (0.85)
  - RC3: test_feynman_transpile[grover_5.qasm]: REL 1.3 s, C30 0.6 s (0.49)
  - RC3: test_feynman_transpile[mod5_4.qasm]: REL 0.3 s, C30 0.3 s (1.00)
  - RC3: test_feynman_transpile[mod_mult_55.qasm]: REL 0.3 s, C30 0.3 s (0.94)
  - RC3: test_feynman_transpile[mod_red_21.qasm]: REL 0.5 s, C30 0.4 s (0.79)
  - RC3: test_feynman_transpile[rb.qasm]: REL 0.1 s, C30 0.1 s (0.88)
  - RC3: ians_transpile[ham_enc_gray_dvalues_4-4-4-4-4-4-4]: REL 1546.5 s, C30 39.1 s (0.03)
  - RC3: hamiltonians_transpile[ham_enc_gray_dvalues_8-8-8]: REL 27.6 s, C30 19.7 s (0.71)
  - RC3: test_hamlib_hamiltonians_transpile[ham_ham_JW-10]: REL 5.3 s, C30 3.8 s (0.71)
  - RC3: test_hamlib_hamiltonians_transpile[ham_ham_JW-14]: REL 40.2 s, C30 62.7 s (1.56)
  - RC3: test_hamlib_hamiltonians_transpile[ham_ham_JW-6]: REL 0.9 s, C30 0.7 s (0.81)
  - RC3: st_hamlib_hamiltonians_transpile[ham_ham_parity10]: REL 16.8 s, C30 12.1 s (0.72)
  - RC3: _transpile[ham_mu_x_prime_enc_unary_dvalues_4-4-4]: REL 0.7 s, C30 0.6 s (0.89)

SUMMARY {"RC0": "PASS", "RC1": "CONFIRMED", "RC2": "CONFIRMED", "RC3": "CONFIRMED"}
