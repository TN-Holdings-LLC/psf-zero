# C35-VAL (pre-registered in Addendum 436)

git head 5ca55be (Addendum 436: pre-registration C35-VAL (lock): should candidate c35 (items 58-63), uncommitted tracked changes: none; 106 tests on FakeTorino, FakeKingston; 1484 jobs, 4 at a time; {'qiskit': '2.5.2', 'qiskit_ibm_runtime': '0.49.0'}; smoke False.

## FakeTorino

106 tests; 106 with every arm; 53 runnable (best ESP >= 0.01).

| arm | returned | errors (FailedElementsError / other) | outputs on failed elements | ESP = 0 on runnable tests | two-qubit / QK2 (gmean, +1) | time summed (s) |
|---|---|---|---|---|---|---|
| QK2 | 106 of 106 | 0 / 0 | 25 | 2 | 1.000 | 60 |
| QK3 | 106 of 106 | 0 / 0 | 22 | 0 | 0.982 | 111 |
| RELR | 106 of 106 | 0 / 0 | 0 | 0 | 1.036 | 448 |
| RELD | 106 of 106 | 0 / 0 | 63 | 20 | 1.066 | 210 |
| C35D | 106 of 106 | 0 / 0 | 0 | 0 | 1.089 | 183 |
| C35R | 106 of 106 | 0 / 0 | 0 | 0 | 1.045 | 354 |
| C35N | 106 of 106 | 0 / 0 | 63 | 20 | 1.066 | 168 |

| ESP ratio (runnable tests, both > 0) | gmean | 10% better | 10% worse | tests |
|---|---|---|---|---|
| C35D / RELR | 1.002 | 5 | 5 | 53 |
| C35D / QK2 | 1.158 | 22 | 2 | 51 |
| C35D / QK3 | 0.957 | 3 | 10 | 53 |
| C35R / RELR | 1.019 | 5 | 4 | 53 |
| C35R / QK3 | 0.973 | 3 | 8 | 53 |
| RELR / QK3 | 0.955 | 1 | 7 | 53 |
| RELD / QK2 | 0.939 | 4 | 32 | 33 |

## FakeKingston

106 tests; 106 with every arm; 65 runnable (best ESP >= 0.01).

| arm | returned | errors (FailedElementsError / other) | outputs on failed elements | ESP = 0 on runnable tests | two-qubit / QK2 (gmean, +1) | time summed (s) |
|---|---|---|---|---|---|---|
| QK2 | 106 of 106 | 0 / 0 | 19 | 2 | 1.000 | 61 |
| QK3 | 106 of 106 | 0 / 0 | 12 | 2 | 0.982 | 138 |
| RELR | 106 of 106 | 0 / 0 | 0 | 0 | 1.023 | 439 |
| RELD | 106 of 106 | 0 / 0 | 17 | 1 | 1.059 | 214 |
| C35D | 106 of 106 | 0 / 0 | 0 | 0 | 1.078 | 187 |
| C35R | 106 of 106 | 0 / 0 | 0 | 0 | 1.031 | 414 |
| C35N | 106 of 106 | 0 / 0 | 17 | 1 | 1.059 | 174 |

| ESP ratio (runnable tests, both > 0) | gmean | 10% better | 10% worse | tests |
|---|---|---|---|---|
| C35D / RELR | 0.964 | 4 | 10 | 65 |
| C35D / QK2 | 1.153 | 20 | 5 | 63 |
| C35D / QK3 | 0.899 | 5 | 17 | 63 |
| C35R / RELR | 0.974 | 4 | 8 | 65 |
| C35R / QK3 | 0.909 | 5 | 16 | 63 |
| RELR / QK3 | 0.933 | 3 | 12 | 63 |
| RELD / QK2 | 0.852 | 6 | 23 | 63 |

## Scoring (Addendum 436)

| ID | FakeTorino | FakeKingston |
|---|---|---|
| V0 | **CONFIRMED**: 0 arm-tests without a circuit | **CONFIRMED**: 0 arm-tests without a circuit |
| V1 | **CONFIRMED**: 0 C35D/C35R outputs on failed elements | **CONFIRMED**: 0 C35D/C35R outputs on failed elements |
| V2 | **CONFIRMED**: ESP C35D / RELR = 1.002 | **CONFIRMED**: ESP C35D / RELR = 0.964 |
| V3 | **CONFIRMED**: ESP C35D / QK2 = 1.158 | **CONFIRMED**: ESP C35D / QK2 = 1.153 |
| V4 | **CONFIRMED**: time C35D / RELR = 183 / 448 s = 0.409 | **CONFIRMED**: time C35D / RELR = 187 / 439 s = 0.427 |
| V5 | **CONFIRMED**: C35N two-qubit = RELD on 106 of 106; summed 1694389 / 1694389 | **CONFIRMED**: C35N two-qubit = RELD on 106 of 106; summed 1688709 / 1688709 |
| V6 | **CONFIRMED**: ESP C35R / RELR = 1.019 | **CONFIRMED**: ESP C35R / RELR = 0.974 |

Release rule (V0, V1, V2, V3, V5 not refuted on either device): **MET: c35 may replace release 2026-10-10.2**

