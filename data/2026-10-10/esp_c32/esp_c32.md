# ESP-C32 (exploratory, nothing predicted)

git head 0b7f790, uncommitted tracked changes: none; 106 tests on FakeTorino, FakeKingston; 424 jobs; {'qiskit': '2.5.2', 'qiskit_ibm_runtime': '0.49.0'}. QK2, QK3 and PSFR (the release's recommended call) are ESP-FT's records.

## FakeTorino

106 tests with every arm; 52 where the best ESP >= 0.01.

| arm | ops on failed elements (tests) | ESP = 0 where the best >= 0.01 | q2 / QK2 (gmean, +1) | time (s, summed) |
|---|---|---|---|---|
| C31D | 0 | 0 | 1.078 | 317 |
| C32D | 0 | 0 | 1.071 | 240 |
| QK2 | 27 | 2 | 1.000 | 60 |
| QK3 | 22 | 0 | 0.976 | 98 |
| PSFR | 0 | 0 | 1.027 | 446 |

ESP ratios where the best ESP >= 0.01 (geometric mean over the tests where both are > 0; wins and losses by 10% or more; 'dead' = the other's ESP is 0 and this arm's is not):

| arm / other | gmean ratio | wins | losses | dead |
|---|---|---|---|---|
| C31D / QK2 | 0.656 | 9 | 23 | 2 |
| C31D / QK3 | 0.562 | 0 | 31 | 0 |
| C31D / PSFR | 0.579 | 1 | 32 | 0 |
| C32D / QK2 | 1.122 | 17 | 5 | 2 |
| C32D / QK3 | 0.948 | 4 | 10 | 0 |
| C32D / PSFR | 0.977 | 0 | 3 | 0 |
| C32D / C31D | 1.689 | 29 | 1 | 0 |

## FakeKingston

106 tests with every arm; 65 where the best ESP >= 0.01.

| arm | ops on failed elements (tests) | ESP = 0 where the best >= 0.01 | q2 / QK2 (gmean, +1) | time (s, summed) |
|---|---|---|---|---|
| C31D | 0 | 0 | 1.060 | 232 |
| C32D | 0 | 0 | 1.059 | 241 |
| QK2 | 15 | 2 | 1.000 | 63 |
| QK3 | 14 | 2 | 0.979 | 108 |
| PSFR | 0 | 0 | 1.018 | 438 |

ESP ratios where the best ESP >= 0.01 (geometric mean over the tests where both are > 0; wins and losses by 10% or more; 'dead' = the other's ESP is 0 and this arm's is not):

| arm / other | gmean ratio | wins | losses | dead |
|---|---|---|---|---|
| C31D / QK2 | 0.835 | 8 | 21 | 2 |
| C31D / QK3 | 0.662 | 2 | 33 | 2 |
| C31D / PSFR | 0.719 | 0 | 31 | 0 |
| C32D / QK2 | 1.161 | 20 | 2 | 2 |
| C32D / QK3 | 0.921 | 3 | 14 | 2 |
| C32D / PSFR | 0.990 | 0 | 2 | 0 |
| C32D / C31D | 1.377 | 30 | 0 | 0 |

