# CANCEL score (SMOKE: not counted)

git_head ceea83c, tests 20 (expected 20), jobs 60
P0 PASS: C23 failed where C22 finished, or invalid: 0; versions ok True; meta ok True
errors or timeouts per arm: {'C22': 0, 'C22B': 0, 'C23': 0}

tests finished by all arms: 20; cancellation tried on 3

| | prediction | value | verdict |
|---|---|---|---|
| K1 | no two-qubit gate cancels, C22 reproduces itself: C23 returns C22's circuit | 0 of 16 differ (1 where C22 did not reproduce) | **CONFIRMED** |
| K2 | some cancel: C23 has no more two-qubit gates than C22 | 0 of 3 have more | **CONFIRMED** |
| K4 | every checkable C23 output implements its input | 0 of 4 do not | **AMBIGUOUS** |
| K5 | no two-qubit gate cancels: median time C23 / C22 <= 1.15 | 1.024 on 17 | **CONFIRMED** |

Reported without prediction: the tests where cancellation was tried

| test | C22 two-qubit | C23 two-qubit | C23 kept | C23 / C22 time |
|---|---|---|---|---|
| test_QASMBench_large[qft_n160-heavy-hex] | 21361 | 15793 | cancelled | 1.54 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-18] | 43491 | 39717 | cancelled | 2.18 |
| test_BVlike_simplification_transpile | 392 | 0 | cancelled | 0.64 |
