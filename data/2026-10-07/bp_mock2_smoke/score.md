# BP-MOCK2 score (SMOKE: not counted)

git_head 7512416, Benchpress b695f30, tests 18 (expected 18), jobs 72
P0 PASS: C22 failed where REL finished, or invalid: 0; versions ok True; meta ok True
errors or timeouts per arm: {'QK': 0, 'REL': 0, 'REL2': 0, 'C22': 0}

tests finished by all arms: 18 (8 flat, 10 wide)

| | prediction | value | verdict |
|---|---|---|---|
| M1 | flat inputs where REL reproduces itself: C22 returns REL's circuit | 0 of 8 differ (0 where REL did not reproduce) | **CONFIRMED** |
| M2 | wide inputs: geometric mean C22/REL two-qubit count <= 0.85 | 0.720 on 10 | **CONFIRMED** |
| M3 | all tests: geometric mean C22/QK two-qubit count <= 1.15 | 1.036 on 18 | **CONFIRMED** |
| M5 | every checkable C22 output implements its input | 0 of 5 do not | **CONFIRMED** |

Reported without prediction:
- REL/QK two-qubit count 1.243; REL2 differs from REL on 1 of 18 tests (two-qubit count differs on 0)
- wide inputs where C22 has more two-qubit gates than REL: []
- REL outputs checked: 5, not implementing: 0; QK outputs not implementing (state check, not comparable after measurement-aware passes): 0

| stratum | tests | C22/QK | REL/QK | C22/REL |
|---|---|---|---|---|
| QASMBench small, all-to-all | 1 | 1.000 | 1.000 | 1.000 |
| QASMBench small, square | 1 | 1.000 | 1.000 | 1.000 |
| QASMBench small, heavy-hex | 1 | 0.969 | 0.969 | 1.000 |
| QASMBench small, linear | 1 | 1.233 | 1.233 | 1.000 |
| QASMBench medium, all-to-all | 1 | 1.000 | 1.000 | 1.000 |
| QASMBench medium, square | 1 | 1.047 | 1.593 | 0.657 |
| QASMBench medium, heavy-hex | 1 | 1.161 | 1.161 | 1.000 |
| QASMBench medium, linear | 1 | 1.000 | 1.000 | 1.000 |
| QASMBench large, all-to-all | 1 | 1.000 | 1.142 | 0.876 |
| QASMBench large, square | 1 | 1.000 | 1.000 | 1.000 |
| QASMBench large, heavy-hex | 1 | 1.009 | 1.009 | 1.000 |
| QASMBench large, linear | 1 | 1.001 | 1.013 | 0.988 |
| HamLib, all-to-all | 1 | 1.000 | 1.999 | 0.500 |
| HamLib, square | 1 | 1.000 | 2.660 | 0.376 |
| HamLib, heavy-hex | 1 | 1.016 | 1.619 | 0.627 |
| HamLib, linear | 1 | 1.009 | 1.201 | 0.840 |
| HamLib, FakeTorino | 1 | 1.254 | 1.882 | 0.666 |
| Feynman, FakeTorino | 1 | 1.000 | 1.000 | 1.000 |
