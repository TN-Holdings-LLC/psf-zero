# KT: the live Target of ibm_kingston (2026-09-28) -- failed elements (exploratory, nothing predicted)

Pickle written by benchmarks/real_target_cliff.py on 2026-09-28 (calibration 2026-09-28 19:55+09:00, as that run printed). 156 qubits; operations ['cz', 'delay', 'id', 'if_else', 'measure', 'measure_2', 'measure_reset', 'measure_reset_2', 'reset', 'reset_2', 'rz', 'sx', 'x', 'xslow']. Failed = reported error >= 0.5.

| operation | entries | error None | median error | entries >= 0.1 | failed (>= 0.5) | error 1 |
|---|---|---|---|---|---|---|
| cz | 352 | 0 | 1.85e-03 | 12 | 8 | 8 |
| delay | 156 | 156 | - | 0 | 0 | 0 |
| id | 156 | 0 | 2.49e-04 | 3 | 3 | 3 |
| measure | 156 | 0 | 1.00e-02 | 4 | 1 | 0 |
| measure_2 | 156 | 0 | 9.40e-03 | 11 | 1 | 0 |
| measure_reset | 156 | 156 | - | 0 | 0 | 0 |
| measure_reset_2 | 156 | 156 | - | 0 | 0 | 0 |
| reset | 156 | 156 | - | 0 | 0 | 0 |
| reset_2 | 156 | 156 | - | 0 | 0 | 0 |
| rz | 156 | 0 | 0.00e+00 | 0 | 0 | 0 |
| sx | 156 | 0 | 2.49e-04 | 3 | 3 | 3 |
| x | 156 | 0 | 2.49e-04 | 3 | 3 | 3 |
| xslow | 156 | 0 | 2.49e-04 | 3 | 3 | 3 |

Couplers (undirected, any two-qubit operation): live 176; FakeKingston (snapshot 2026-04) 176; in the snapshot but not in the live Target: []; in the live Target but not in the snapshot: [].
Qubits with no operation at all in the live Target: [].

Failed cz: [112, 113] (1), [113, 112] (1), [130, 131] (1), [131, 130] (1), [145, 146] (1), [146, 145] (1), [146, 147] (1), [147, 146] (1)
Failed id: [113] (1), [121] (1), [146] (1)
Failed measure: [146] (0.504)
Failed measure_2: [146] (0.504)
Failed sx: [113] (1), [121] (1), [146] (1)
Failed x: [113] (1), [121] (1), [146] (1)
Failed xslow: [113] (1), [121] (1), [146] (1)

## Answer

Yes: the live Target keeps 8 two-qubit entries, 13 one-qubit gate entries and 1 measurements with error >= 0.5. A compiler that reads only the coupling map can place gates on them; ESP prices such an output at (about) 0.
