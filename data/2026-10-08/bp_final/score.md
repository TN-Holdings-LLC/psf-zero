# BP-FINAL score

git_head 79b73ed, Benchpress b695f30, Python 3.11.9, Qiskit 2.5.2, tests 880 (expected 880), jobs 2746, 13354.8 s
P0 PASS: every arm present True; versions True; release, reference and Benchpress as locked, no uncommitted change True
errors or timeouts per arm: {'QK': 2, 'REL': 3, 'REL2': 3, 'RECR': 1}

tests finished by QK and REL: 877 of 880

| | prediction | value | verdict |
|---|---|---|---|
| F1 | all tests: geometric mean REL/QK two-qubit count <= 1.06 (REFUTED > 1.10) | 1.046 (95% 1.037-1.055) on 877 | **CONFIRMED** |
| F2 | every family <= 1.15 (REFUTED if any > 1.25) | QASMBench 1.020 (404); HamLib, abstract 1.069 (367); HamLib, FakeTorino 1.090 (67); Feynman, FakeTorino 1.022 (39) | **CONFIRMED** |
| F3 | share of tests with more than 1.10 times QK's count <= 0.15 (REFUTED > 0.25) | 0.149 (131 of 877) | **CONFIRMED** |
| F4 | every REL, REL2 and RECR output passes Benchpress's validator, and every checkable one implements its input (at least 20 checked) | 0 invalid of 1859; 0 not implementing of 491 | **CONFIRMED** |
| F5 | the recommended call places no two-qubit gate on a failed coupler or qubit of FakeTorino | 0 of 105 tests | **CONFIRMED** |
| F6 | REL fails (error or timeout) on <= 1% of the tests QK finishes (REFUTED > 3%) | 0.001 (1 of 878) | **CONFIRMED** |

Reported without prediction:
- two-qubit depth REL/QK 1.040; families weighted equally 1.050
- fewer two-qubit gates than QK on 117, as many on 314, more on 446
- compile time REL/QK: median 3.52, geometric mean 2.99 (jobs run 12 at a time)
- REL2 differs from REL on 83 of 877 tests (two-qubit count on 0)
- FakeTorino: RECR/QK two-qubit count 1.035 on 105; tests with gates on failed elements: QK 25, REL 62, RECR 0
- REL outputs checked for equivalence: 231

| stratum | tests | REL/QK | REL/QK depth | time REL/QK (median) |
|---|---|---|---|---|
| QASMBench small, all-to-all | 32 | 1.005 | 1.005 | 6.20 |
| QASMBench small, square | 34 | 1.015 | 1.016 | 6.44 |
| QASMBench small, heavy-hex | 34 | 1.020 | 0.993 | 4.48 |
| QASMBench small, linear | 34 | 1.023 | 1.021 | 6.47 |
| QASMBench medium, all-to-all | 17 | 1.002 | 1.000 | 4.05 |
| QASMBench medium, square | 15 | 1.051 | 1.024 | 4.20 |
| QASMBench medium, heavy-hex | 17 | 1.028 | 1.000 | 4.52 |
| QASMBench medium, linear | 17 | 1.007 | 1.004 | 4.41 |
| QASMBench large, all-to-all | 52 | 1.020 | 1.000 | 3.04 |
| QASMBench large, square | 51 | 1.056 | 1.036 | 2.74 |
| QASMBench large, heavy-hex | 51 | 1.027 | 1.024 | 2.88 |
| QASMBench large, linear | 50 | 0.987 | 0.983 | 2.75 |
| HamLib, all-to-all | 92 | 1.059 | 1.060 | 3.92 |
| HamLib, square | 92 | 1.082 | 1.081 | 2.88 |
| HamLib, heavy-hex | 91 | 1.078 | 1.070 | 2.58 |
| HamLib, linear | 92 | 1.056 | 1.054 | 2.27 |
| HamLib, FakeTorino | 67 | 1.090 | 1.106 | 2.91 |
| Feynman, FakeTorino | 39 | 1.022 | 1.025 | 5.22 |

SUMMARY {"F1": "CONFIRMED", "F2": "CONFIRMED", "F3": "CONFIRMED", "F4": "CONFIRMED", "F5": "CONFIRMED", "F6": "CONFIRMED", "P0": "PASS", "F1_value": 1.045521}
