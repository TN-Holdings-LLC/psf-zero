# ROUTE-X (exploratory)

git head 3e18265 (Addendum 438: release psf_compile 2026-10-10.3 = candidate c35 (items 58-63: bac); 106 tests on FakeTorino, FakeKingston; 1060 jobs.

## FakeTorino

106 tests; 106 with every arm; 54 runnable.

| arm | returned | outputs on failed elements | ESP = 0 on runnable | two-qubit / QK3 (gmean, +1) | time summed (s) | median time (s) |
|---|---|---|---|---|---|---|
| R1 | 106 of 106 | 0 | 0 | 1.108 | 192 | 0.172 |
| R2 | 106 of 106 | 0 | 0 | 1.031 | 202 | 0.219 |
| R3 | 106 of 106 | 0 | 0 | 1.011 | 222 | 0.203 |
| QK2 | 106 of 106 | 25 | 2 | 1.021 | 65 | 0.079 |
| QK3 | 106 of 106 | 24 | 0 | 1.000 | 114 | 0.151 |

| ESP ratio, runnable tests, both > 0 | gmean | 10% better | 10% worse | tests |
|---|---|---|---|---|
| R1 / QK3 | 0.929 | 4 | 12 | 54 |
| R2 / QK3 | 0.956 | 2 | 9 | 54 |
| R3 / QK3 | 0.975 | 2 | 6 | 54 |
| R3 / R1 | 1.050 | 9 | 2 | 54 |
| R2 / R1 | 1.029 | 8 | 3 | 54 |
| R1 / QK2 | 1.167 | 20 | 3 | 52 |
| R3 / QK2 | 1.227 | 21 | 4 | 52 |

## FakeKingston

106 tests; 106 with every arm; 65 runnable.

| arm | returned | outputs on failed elements | ESP = 0 on runnable | two-qubit / QK3 (gmean, +1) | time summed (s) | median time (s) |
|---|---|---|---|---|---|---|
| R1 | 106 of 106 | 0 | 0 | 1.094 | 205 | 0.188 |
| R2 | 106 of 106 | 0 | 0 | 1.023 | 196 | 0.257 |
| R3 | 106 of 106 | 0 | 0 | 1.007 | 246 | 0.242 |
| QK2 | 106 of 106 | 14 | 1 | 1.018 | 63 | 0.094 |
| QK3 | 106 of 106 | 14 | 2 | 1.000 | 139 | 0.127 |

| ESP ratio, runnable tests, both > 0 | gmean | 10% better | 10% worse | tests |
|---|---|---|---|---|
| R1 / QK3 | 0.897 | 3 | 17 | 63 |
| R2 / QK3 | 0.960 | 4 | 11 | 63 |
| R3 / QK3 | 0.985 | 5 | 7 | 63 |
| R3 / R1 | 1.102 | 15 | 0 | 65 |
| R2 / R1 | 1.070 | 12 | 2 | 65 |
| R1 / QK2 | 1.119 | 19 | 8 | 64 |
| R3 / QK2 | 1.236 | 25 | 4 | 64 |

