# BP-MOCK2 score

git_head fffedd5, Benchpress b695f30, tests 48 (expected 48), jobs 192
P0 PASS: C22 failed where REL finished, or invalid: 0; versions ok True; meta ok True
errors or timeouts per arm: {'QK': 1, 'REL': 1, 'REL2': 1, 'C22': 0}

tests finished by all arms: 47 (17 flat, 30 wide)

| | prediction | value | verdict |
|---|---|---|---|
| M1 | flat inputs where REL reproduces itself: C22 returns REL's circuit | 0 of 16 differ (1 where REL did not reproduce) | **CONFIRMED** |
| M2 | wide inputs: geometric mean C22/REL two-qubit count <= 0.85 | 0.738 on 30 | **CONFIRMED** |
| M3 | all tests: geometric mean C22/QK two-qubit count <= 1.15 | 1.039 on 47 | **CONFIRMED** |
| M5 | every checkable C22 output implements its input | 0 of 15 do not | **CONFIRMED** |

Reported without prediction:
- REL/QK two-qubit count 1.261; REL2 differs from REL on 2 of 47 tests (two-qubit count differs on 0)
- wide inputs where C22 has more two-qubit gates than REL: ['test_QASMBench_large[qugan_n39-heavy-hex]', 'test_feynman_transpile[grover_5.qasm]', 'test_feynman_transpile[hwb10.qasm]']
- REL outputs checked: 15, not implementing: 0; QK outputs not implementing (state check, not comparable after measurement-aware passes): 2

| stratum | tests | C22/QK | REL/QK | C22/REL |
|---|---|---|---|---|
| QASMBench small, all-to-all | 3 | 1.000 | 1.000 | 1.000 |
| QASMBench small, square | 3 | 1.084 | 1.133 | 0.956 |
| QASMBench small, heavy-hex | 3 | 1.014 | 1.014 | 1.000 |
| QASMBench small, linear | 3 | 1.076 | 1.091 | 0.986 |
| QASMBench medium, all-to-all | 2 | 1.000 | 1.000 | 1.000 |
| QASMBench medium, square | 2 | 1.044 | 1.288 | 0.810 |
| QASMBench medium, heavy-hex | 2 | 1.153 | 1.158 | 0.995 |
| QASMBench medium, linear | 2 | 1.020 | 1.342 | 0.760 |
| QASMBench large, all-to-all | 2 | 1.000 | 1.069 | 0.936 |
| QASMBench large, square | 2 | 1.000 | 1.000 | 1.000 |
| QASMBench large, heavy-hex | 2 | 1.083 | 1.073 | 1.010 |
| QASMBench large, linear | 1 | 1.004 | 1.017 | 0.988 |
| HamLib, all-to-all | 3 | 0.998 | 1.268 | 0.787 |
| HamLib, square | 3 | 1.015 | 2.048 | 0.496 |
| HamLib, heavy-hex | 3 | 1.020 | 2.337 | 0.437 |
| HamLib, linear | 3 | 1.046 | 1.551 | 0.674 |
| HamLib, FakeTorino | 4 | 1.095 | 1.473 | 0.744 |
| Feynman, FakeTorino | 4 | 1.035 | 1.073 | 0.964 |
