# BP-MOCK score

git_head 86f992c, Benchpress b695f30, tests 92 (expected 92), jobs 316, flags []
P0 PASS: C22 failed where REL finished 0, C22 invalid 0, C22R failed or invalid 0
errors or timeouts per arm: {'QK': 0, 'REL': 0, 'C22': 0, 'RELR': 0, 'C22R': 0}

tests finished by QK, REL and C22: 92 (38 without wide instructions, 54 with); C22R and RELR both finished: 20; equivalence checked on 14

| | prediction | value | verdict |
|---|---|---|---|
| M1 | inputs without wide instructions: C22's default call returns REL's circuit | 2 of 38 differ | **REFUTED** |
| M2 | inputs with wide instructions: geometric mean C22/REL two-qubit count <= 0.85 | 0.709 | **CONFIRMED** |
| M3 | all tests: geometric mean C22/QK two-qubit count <= 1.15 | 1.115 | **CONFIRMED** |
| M4 | FakeTorino tests: geometric mean C22R/RELR two-qubit count <= 1.00 | 0.887 | **CONFIRMED** |
| M5 | every C22 output checked for equivalence (<= 10 qubits) is equivalent | 5 of 14 not equivalent | **REFUTED** |

Reported without prediction:
- REL/QK two-qubit count, all tests: 1.364; two-qubit depth C22/QK 1.117, REL/QK 1.279

| stratum | tests | C22/QK q2 | REL/QK q2 | C22/QK d2 | C22 time / QK time |
|---|---|---|---|---|---|
| QASMBench small, all-to-all | 5 | 1.095 | 1.095 | 1.070 | 9.29 |
| QASMBench small, square | 5 | 1.063 | 1.063 | 1.026 | 10.74 |
| QASMBench small, heavy-hex | 5 | 1.006 | 1.106 | 1.009 | 7.41 |
| QASMBench small, linear | 5 | 0.956 | 0.956 | 0.956 | 6.85 |
| QASMBench medium, all-to-all | 4 | 1.000 | 1.020 | 1.000 | 4.26 |
| QASMBench medium, square | 4 | 1.011 | 1.070 | 0.981 | 4.51 |
| QASMBench medium, heavy-hex | 4 | 1.050 | 1.446 | 1.031 | 2.95 |
| QASMBench medium, linear | 4 | 1.040 | 1.064 | 1.123 | 3.08 |
| QASMBench large, all-to-all | 4 | 1.000 | 1.068 | 1.000 | 2.16 |
| QASMBench large, square | 4 | 1.032 | 1.263 | 1.043 | 2.78 |
| QASMBench large, heavy-hex | 4 | 1.119 | 1.447 | 1.082 | 0.95 |
| QASMBench large, linear | 4 | 0.992 | 1.555 | 0.990 | 3.07 |
| HamLib, all-to-all | 5 | 1.004 | 1.297 | 1.004 | 4.00 |
| HamLib, square | 5 | 1.048 | 1.868 | 1.065 | 3.12 |
| HamLib, heavy-hex | 5 | 1.041 | 1.452 | 1.029 | 1.83 |
| HamLib, linear | 5 | 1.218 | 1.464 | 1.244 | 2.24 |
| HamLib, FakeTorino | 8 | 1.119 | 1.581 | 1.111 | 2.90 |
| Feynman, FakeTorino | 6 | 1.008 | 1.261 | 1.066 | 5.32 |
| 100-qubit, FakeTorino | 6 | 2.767 | 3.630 | 2.773 | 1.62 |

- QK outputs not equivalent: 7 of 14; REL: 5 of 14
- C22R/RELR time: 1.15 (geometric mean); C22R/QK two-qubit count 1.673
