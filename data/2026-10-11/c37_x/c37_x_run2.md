# C37-X (exploratory; same tests and devices as GAP-DIAG)

jobs with an error: 0

## Failed elements (outputs with an operation at error >= 0.5)

| device | QK3 | REL | C37 |
|---|---|---|---|
| FakeTorino | 22 | 0 | 0 |
| FakeKingston | 10 | 0 | 0 |
| FakeMarrakesh | 28 | 0 | 0 |
| FakeFez | 28 | 0 | 0 |

## Runnable tests (best ESP of the three >= 0.01), all outputs clean: geometric-mean ESP ratios and mean part differences (log10; positive = the first arm loses)

| device | tests | C37/QK3 | REL/QK3 | C37/REL | 2q part C37-QK3 | 1q part C37-QK3 | meas part C37-QK3 | 1q part C37-REL | meas part C37-REL | non-rz 1q C37/REL/QK3 | time C37/REL/QK3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FakeTorino | 53 | 0.994 | 0.977 | 1.017 | +0.00358 | -0.00201 | +0.00108 | -0.00345 | -0.00406 | 19765/21352/20623 | 11/9/5 |
| FakeKingston | 64 | 1.016 | 0.997 | 1.019 | -0.00587 | -0.00109 | +0.00006 | -0.00552 | -0.00270 | 52253/56040/53457 | 23/17/10 |
| FakeMarrakesh | 56 | 1.032 | 1.023 | 1.009 | -0.00974 | -0.00268 | -0.00116 | -0.00346 | -0.00031 | 28616/30993/29250 | 16/13/6 |
| FakeFez | 55 | 1.020 | 1.012 | 1.008 | -0.00663 | -0.00241 | +0.00050 | -0.00286 | -0.00056 | 24792/26454/24885 | 15/13/8 |

## Essential: runnable tests where QK3's own failed elements count against it (ESP of a failed output = 0)

| device | runnable | QK3 ESP 0 | C37 ESP 0 | C37/QK3 on both > 0 |
|---|---|---|---|---|
| FakeTorino | 53 | 0 | 0 | 0.994 |
| FakeKingston | 65 | 1 | 0 | 1.016 |
| FakeMarrakesh | 56 | 0 | 0 | 1.032 |
| FakeFez | 55 | 0 | 0 | 1.020 |

## Largest C37 losses to QK3 (clean runnable tests, log10 ESP)

- FakeFez test_feynman_transpile[ham15-low.qasm]: -0.2040 (2q 787 vs 781; parts C37 1.287/0.159/0.000, QK3 1.108/0.135/0.000)
- FakeTorino rid-pbc-qubitnodes_Lx-14_U-10_enc-stdbinary_d-4]: -0.1921 (2q 1058 vs 961; parts C37 1.761/0.255/0.000, QK3 1.566/0.259/0.000)
- FakeKingston rid-pbc-qubitnodes_Lx-24_U-20_enc-stdbinary_d-4]: -0.1327 (2q 1823 vs 1791; parts C37 1.481/0.400/0.000, QK3 1.387/0.362/0.000)
- FakeTorino test_feynman_transpile[qcla_mod_7.qasm]: -0.1204 (2q 777 vs 868; parts C37 1.419/0.183/0.000, QK3 1.271/0.210/0.000)
- FakeTorino test_feynman_transpile[tof_10.qasm]: -0.0961 (2q 186 vs 143; parts C37 0.296/0.044/0.000, QK3 0.207/0.036/0.000)
- FakeMarrakesh test_feynman_transpile[tof_10.qasm]: -0.0898 (2q 188 vs 145; parts C37 0.194/0.041/0.000, QK3 0.121/0.025/0.000)
- FakeKingston transpile[ham_tsp_prob-lin105_Ncity-7_enc-unary]: -0.0809 (2q 1667 vs 1570; parts C37 1.334/0.331/0.000, QK3 1.276/0.308/0.000)
- FakeFez test_feynman_transpile[adder_8.qasm]: -0.0743 (2q 782 vs 736; parts C37 1.118/0.147/0.000, QK3 1.053/0.137/0.000)
- FakeMarrakesh ltonians_transpile[ham_complbipart-n-10_a-5_b-5]: -0.0674 (2q 290 vs 291; parts C37 0.298/0.060/0.000, QK3 0.247/0.044/0.000)
- FakeFez test_feynman_transpile[tof_10.qasm]: -0.0672 (2q 194 vs 143; parts C37 0.264/0.033/0.000, QK3 0.201/0.029/0.000)
- FakeKingston rid-pbc-qubitnodes_Lx-14_U-10_enc-stdbinary_d-4]: -0.0631 (2q 1058 vs 1051; parts C37 0.763/0.209/0.000, QK3 0.692/0.217/0.000)
- FakeMarrakesh test_feynman_transpile[gf2^7_mult.qasm]: -0.0586 (2q 803 vs 781; parts C37 1.025/0.225/0.000, QK3 0.985/0.207/0.000)

FINAL_STATS summed over C37 calls: {'push_calls': 564, 'push_applied': 520, 'relabel_calls': 564, 'relabel_applied': 8}
