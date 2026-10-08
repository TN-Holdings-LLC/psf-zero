# BP-FINAL score (SMOKE: tests already used, not counted)

git_head ba4f2d0, Benchpress b695f30, Python 3.11.9, Qiskit 2.5.2, tests 18 (expected 18), jobs 56, 163.4 s
P0 PASS: every arm present True; versions True; release, reference and Benchpress as locked, no uncommitted change True
errors or timeouts per arm: {'QK': 0, 'REL': 0, 'REL2': 0, 'RECR': 0}

tests finished by QK and REL: 18 of 18

| | prediction | value | verdict |
|---|---|---|---|
| F1 | all tests: geometric mean REL/QK two-qubit count <= 1.06 (REFUTED > 1.10) | 1.016 (95% 1.016-1.016) on 18 | **CONFIRMED** |
| F2 | every family <= 1.15 (REFUTED if any > 1.25) | QASMBench 1.017 (12); HamLib, abstract 1.009 (4); HamLib, FakeTorino 1.056 (1); Feynman, FakeTorino 1.000 (1) | **CONFIRMED** |
| F3 | share of tests with more than 1.10 times QK's count <= 0.15 (REFUTED > 0.25) | 0.056 (1 of 18) | **CONFIRMED** |
| F4 | every REL, REL2 and RECR output passes Benchpress's validator, and every checkable one implements its input (at least 20 checked) | 0 invalid of 38; 0 not implementing of 11 | **AMBIGUOUS** |
| F5 | the recommended call places no two-qubit gate on a failed coupler or qubit of FakeTorino | 0 of 2 tests | **CONFIRMED** |
| F6 | REL fails (error or timeout) on <= 1% of the tests QK finishes (REFUTED > 3%) | 0.000 (0 of 18) | **CONFIRMED** |

Reported without prediction:
- two-qubit depth REL/QK 1.019; families weighted equally 1.020
- fewer two-qubit gates than QK on 1, as many on 9, more on 8
- compile time REL/QK: median 4.85, geometric mean 3.33 (jobs run 12 at a time)
- REL2 differs from REL on 2 of 18 tests (two-qubit count on 0)
- FakeTorino: RECR/QK two-qubit count 1.028 on 2; tests with gates on failed elements: QK 0, REL 1, RECR 0
- REL outputs checked for equivalence: 5

| stratum | tests | REL/QK | REL/QK depth | time REL/QK (median) |
|---|---|---|---|---|
| QASMBench small, all-to-all | 1 | 1.000 | 1.000 | 7.95 |
| QASMBench small, square | 1 | 1.000 | 1.000 | 6.09 |
| QASMBench small, heavy-hex | 1 | 1.033 | 0.959 | 5.86 |
| QASMBench small, linear | 1 | 1.156 | 1.067 | 4.83 |
| QASMBench medium, all-to-all | 1 | 1.000 | 1.000 | 4.87 |
| QASMBench medium, square | 1 | 0.975 | 1.000 | 7.35 |
| QASMBench medium, heavy-hex | 1 | 1.029 | 1.095 | 6.82 |
| QASMBench medium, linear | 1 | 1.000 | 1.000 | 6.89 |
| QASMBench large, all-to-all | 1 | 1.000 | 1.000 | 4.42 |
| QASMBench large, square | 1 | 1.000 | 1.000 | 9.52 |
| QASMBench large, heavy-hex | 1 | 1.014 | 1.209 | 0.16 |
| QASMBench large, linear | 1 | 1.003 | 0.990 | 0.78 |
| HamLib, all-to-all | 1 | 1.000 | 1.000 | 1.37 |
| HamLib, square | 1 | 1.000 | 1.000 | 4.22 |
| HamLib, heavy-hex | 1 | 1.016 | 1.000 | 7.24 |
| HamLib, linear | 1 | 1.020 | 1.008 | 1.21 |
| HamLib, FakeTorino | 1 | 1.056 | 1.031 | 3.53 |
| Feynman, FakeTorino | 1 | 1.000 | 1.000 | 1.13 |

SUMMARY {"F1": "CONFIRMED", "F2": "CONFIRMED", "F3": "CONFIRMED", "F4": "AMBIGUOUS", "F5": "CONFIRMED", "F6": "CONFIRMED", "P0": "PASS", "F1_value": 1.016113}
