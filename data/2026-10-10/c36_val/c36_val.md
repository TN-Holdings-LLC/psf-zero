# C36-VAL (pre-registered in Addendum 440)

git head 866fea6 (Addendum 440: pre-registration C36-VAL (lock): candidate c36 (item 64, routing a), uncommitted tracked changes: none; 106 tests on FakeMarrakesh, FakeFez; 1696 jobs, 4 at a time; {'qiskit': '2.5.2', 'qiskit_ibm_runtime': '0.49.0'}; smoke False.

## FakeMarrakesh

106 tests; 106 with every arm; 57 runnable (best ESP >= 0.01).

| arm | returned | errors (FailedElementsError / other) | outputs on failed elements | ESP = 0 on runnable tests | two-qubit / QK2 (gmean, +1) | time summed (s) |
|---|---|---|---|---|---|---|
| QK2 | 106 of 106 | 0 / 0 | 32 | 1 | 1.000 | 67 |
| QK3 | 106 of 106 | 0 / 0 | 29 | 0 | 0.966 | 148 |
| REL3D | 106 of 106 | 0 / 0 | 0 | 0 | 1.075 | 186 |
| REL3R | 106 of 106 | 0 / 0 | 0 | 0 | 1.035 | 400 |
| REL3N | 106 of 106 | 0 / 0 | 52 | 15 | 1.046 | 180 |
| C36D | 106 of 106 | 0 / 0 | 0 | 0 | 0.989 | 233 |
| C36R | 106 of 106 | 0 / 0 | 0 | 0 | 0.990 | 432 |
| C36N | 106 of 106 | 0 / 0 | 52 | 15 | 1.046 | 184 |

| ESP ratio (runnable tests, both > 0) | gmean | 10% better | 10% worse | tests |
|---|---|---|---|---|
| C36D / REL3D | 1.059 | 9 | 2 | 57 |
| C36D / QK3 | 0.976 | 3 | 8 | 57 |
| C36D / QK2 | 0.989 | 8 | 7 | 56 |
| C36R / REL3R | 1.048 | 9 | 2 | 57 |
| C36R / QK3 | 0.984 | 3 | 6 | 57 |
| REL3D / QK3 | 0.922 | 3 | 13 | 57 |
| REL3R / QK3 | 0.939 | 3 | 9 | 57 |

## FakeFez

106 tests; 106 with every arm; 55 runnable (best ESP >= 0.01).

| arm | returned | errors (FailedElementsError / other) | outputs on failed elements | ESP = 0 on runnable tests | two-qubit / QK2 (gmean, +1) | time summed (s) |
|---|---|---|---|---|---|---|
| QK2 | 106 of 106 | 0 / 0 | 28 | 1 | 1.000 | 66 |
| QK3 | 106 of 106 | 0 / 0 | 28 | 0 | 0.975 | 143 |
| REL3D | 106 of 106 | 0 / 0 | 0 | 0 | 1.087 | 182 |
| REL3R | 106 of 106 | 0 / 0 | 0 | 0 | 1.045 | 398 |
| REL3N | 106 of 106 | 0 / 0 | 61 | 14 | 1.052 | 173 |
| C36D | 106 of 106 | 0 / 0 | 0 | 0 | 0.998 | 244 |
| C36R | 106 of 106 | 0 / 0 | 0 | 0 | 0.993 | 433 |
| C36N | 106 of 106 | 0 / 0 | 61 | 14 | 1.052 | 173 |

| ESP ratio (runnable tests, both > 0) | gmean | 10% better | 10% worse | tests |
|---|---|---|---|---|
| C36D / REL3D | 1.077 | 9 | 3 | 55 |
| C36D / QK3 | 0.963 | 2 | 9 | 55 |
| C36D / QK2 | 1.087 | 12 | 2 | 54 |
| C36R / REL3R | 1.062 | 7 | 3 | 55 |
| C36R / QK3 | 0.969 | 2 | 8 | 55 |
| REL3D / QK3 | 0.894 | 1 | 11 | 55 |
| REL3R / QK3 | 0.912 | 2 | 8 | 55 |

## Scoring (Addendum 440)

| ID | FakeMarrakesh | FakeFez |
|---|---|---|
| W0 | **CONFIRMED**: 0 arm-tests without a circuit | **CONFIRMED**: 0 arm-tests without a circuit |
| W1 | **CONFIRMED**: 0 C36D/C36R outputs on failed elements | **CONFIRMED**: 0 C36D/C36R outputs on failed elements |
| W2 | **CONFIRMED**: ESP C36D / REL3D = 1.059 | **CONFIRMED**: ESP C36D / REL3D = 1.077 |
| W3 | **CONFIRMED**: ESP C36D / QK3 = 0.976 | **CONFIRMED**: ESP C36D / QK3 = 0.963 |
| W4 | **CONFIRMED**: C36N two-qubit = REL3N on 106 of 106 | **CONFIRMED**: C36N two-qubit = REL3N on 106 of 106 |
| W5 | **CONFIRMED**: time C36D / REL3D = 233 / 186 s = 1.255 | **CONFIRMED**: time C36D / REL3D = 244 / 182 s = 1.340 |
| W6 | **CONFIRMED**: ESP C36R / REL3R = 1.048 | **CONFIRMED**: ESP C36R / REL3R = 1.062 |

Release rule (W0, W1, W2, W4 not refuted on either device): **MET: c36 may replace release 2026-10-10.3**

