# BP-MOCK score (SMOKE: not counted)

git_head 6dbadfa, Benchpress b695f30, tests 19 (expected 19), jobs 63, flags []
P0 PASS: C22 failed where REL finished 0, C22 invalid 0, C22R failed or invalid 0
errors or timeouts per arm: {'QK': 0, 'REL': 0, 'C22': 0, 'RELR': 0, 'C22R': 0}

tests finished by QK, REL and C22: 19 (7 without wide instructions, 12 with); C22R and RELR both finished: 3; equivalence checked on 3

| | prediction | value | verdict |
|---|---|---|---|
| M1 | inputs without wide instructions: C22's default call returns REL's circuit | 0 of 7 differ | **CONFIRMED** |
| M2 | inputs with wide instructions: geometric mean C22/REL two-qubit count <= 0.85 | 0.672 | **CONFIRMED** |
| M3 | all tests: geometric mean C22/QK two-qubit count <= 1.15 | 1.087 | **CONFIRMED** |
| M4 | FakeTorino tests: geometric mean C22R/RELR two-qubit count <= 1.00 | 0.837 | **CONFIRMED** |
| M5 | every C22 output checked for equivalence (<= 10 qubits) is equivalent | 0 of 3 not equivalent | **CONFIRMED** |

Reported without prediction:
- REL/QK two-qubit count, all tests: 1.397; two-qubit depth C22/QK 1.068, REL/QK 1.298

| stratum | tests | C22/QK q2 | REL/QK q2 | C22/QK d2 | C22 time / QK time |
|---|---|---|---|---|---|
| QASMBench small, all-to-all | 1 | 1.571 | 1.571 | 1.400 | 25.28 |
| QASMBench small, square | 1 | 1.000 | 1.000 | 1.000 | 14.34 |
| QASMBench small, heavy-hex | 1 | 1.000 | 1.000 | 1.000 | 9.90 |
| QASMBench small, linear | 1 | 1.000 | 1.000 | 1.000 | 2.35 |
| QASMBench medium, all-to-all | 1 | 1.000 | 1.000 | 1.000 | 3.73 |
| QASMBench medium, square | 1 | 1.012 | 1.269 | 0.724 | 6.83 |
| QASMBench medium, heavy-hex | 1 | 1.129 | 1.753 | 1.182 | 9.08 |
| QASMBench medium, linear | 1 | 1.053 | 1.152 | 1.109 | 2.94 |
| QASMBench large, all-to-all | 1 | 1.000 | 1.091 | 1.000 | 2.15 |
| QASMBench large, square | 1 | 1.126 | 1.481 | 1.121 | 3.80 |
| QASMBench large, heavy-hex | 1 | 1.436 | 1.436 | 1.438 | 2.43 |
| QASMBench large, linear | 1 | 1.000 | 1.000 | 1.000 | 3.78 |
| HamLib, all-to-all | 1 | 1.000 | 1.071 | 1.000 | 1.89 |
| HamLib, square | 1 | 1.000 | 1.939 | 1.000 | 8.16 |
| HamLib, heavy-hex | 1 | 1.065 | 1.663 | 0.985 | 0.75 |
| HamLib, linear | 1 | 1.316 | 1.912 | 1.316 | 10.96 |
| HamLib, FakeTorino | 1 | 1.158 | 1.547 | 1.132 | 1.08 |
| Feynman, FakeTorino | 1 | 0.985 | 1.178 | 1.112 | 5.67 |
| 100-qubit, FakeTorino | 1 | 1.000 | 5.104 | 1.000 | 0.71 |

- QK outputs not equivalent: 1 of 3; REL: 0 of 3
- C22R/RELR time: 1.92 (geometric mean); C22R/QK two-qubit count 1.629
