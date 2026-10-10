# REPL-C33 (exploratory, nothing predicted)

git head 0b7f790, uncommitted tracked changes: none; 106 tests on FakeTorino, FakeKingston; 636 jobs, 4 at a time; Qiskit 2.5.2. Cold: the first call in the process; warm: the second.

## FakeTorino

106 of 106 tests with every arm; failed jobs: QK2 0, C32D 0, C33D 0

| arm | time cold (s, summed) | time warm (s, summed) | cold / QK2 cold (gmean) | warm / QK2 warm (gmean) | small tests: median cold / warm (s) | q2 / QK2 (gmean, +1) | outputs on failed elements |
|---|---|---|---|---|---|---|---|
| QK2 | 63 | 60 | 1.00 | 1.00 | 0.076 / 0.039 | 1.000 | 25 |
| C32D | 256 | 235 | 3.95 | 3.24 | 0.259 / 0.139 | 1.073 | 0 |
| C33D | 218 | 213 | 3.49 | 2.93 | 0.248 / 0.125 | 1.082 | 0 |

- C33D compiled once on the pruned map on 106 tests; C32D recompiled on 24.
- C33D / C32D cold time: geometric mean 0.882, median 0.937; the same output by value on 45 of 106.
- ESP where the best arm's >= 0.01 (52 tests): C33D / C32D geometric mean 1.019; 10% better on 7, 10% worse on 4.

## FakeKingston

106 of 106 tests with every arm; failed jobs: QK2 0, C32D 0, C33D 0

| arm | time cold (s, summed) | time warm (s, summed) | cold / QK2 cold (gmean) | warm / QK2 warm (gmean) | small tests: median cold / warm (s) | q2 / QK2 (gmean, +1) | outputs on failed elements |
|---|---|---|---|---|---|---|---|
| QK2 | 80 | 59 | 1.00 | 1.00 | 0.251 / 0.044 | 1.000 | 13 |
| C32D | 257 | 220 | 2.40 | 3.32 | 0.490 / 0.124 | 1.060 | 0 |
| C33D | 249 | 210 | 2.22 | 2.96 | 0.493 / 0.121 | 1.074 | 0 |

- C33D compiled once on the pruned map on 106 tests; C32D recompiled on 15.
- C33D / C32D cold time: geometric mean 0.922, median 0.968; the same output by value on 44 of 106.
- ESP where the best arm's >= 0.01 (64 tests): C33D / C32D geometric mean 0.969; 10% better on 2, 10% worse on 8.

