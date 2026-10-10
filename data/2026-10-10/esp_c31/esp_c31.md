# ESP-C31 (exploratory, nothing predicted)

git head 0b7f790, uncommitted tracked changes: none; 106 tests on FakeTorino, FakeKingston; 636 jobs; {'qiskit': '2.5.2', 'qiskit_ibm_runtime': '0.49.0'}

## FakeTorino

| arm | outputs | refused (FailedElementsError) | other errors | outputs with ops on failed elements | ops on failed elements | q2 / RELT (gmean, +1, both returned) | time (s, summed) |
|---|---|---|---|---|---|---|---|
| RELT | 106 | 0 | 0 | 0 | 0 | 1.000 | 319 |
| C31D | 106 | 0 | 0 | 0 | 0 | 1.000 | 318 |
| C31R | 106 | 0 | 0 | 0 | 0 | 0.953 | 435 |

- C31D's two-qubit count equals RELT's on 106 of 106 tests where both returned
- C31R's two-qubit count equals the release's recommended call (ESP-FT) on 106 of 106

## FakeKingston

| arm | outputs | refused (FailedElementsError) | other errors | outputs with ops on failed elements | ops on failed elements | q2 / RELT (gmean, +1, both returned) | time (s, summed) |
|---|---|---|---|---|---|---|---|
| RELT | 106 | 0 | 0 | 0 | 0 | 1.000 | 243 |
| C31D | 106 | 0 | 0 | 0 | 0 | 1.000 | 242 |
| C31R | 106 | 0 | 0 | 0 | 0 | 0.961 | 443 |

- C31D's two-qubit count equals RELT's on 106 of 106 tests where both returned
- C31R's two-qubit count equals the release's recommended call (ESP-FT) on 106 of 106

