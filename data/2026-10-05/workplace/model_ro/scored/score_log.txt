# MODEL-RO score

P0: FAIL -- devices 3/3, circuits per device [153, 153, 153] (expected 153); max noiseless total variation 4.77e-06 (<= 1e-6); clbit j measures logical j's final qubit: True

| device | arm | classical infidelity | summed measure error | 2q | median compile s |
|---|---|---|---|---|---|
| FakeTorino | A9 | 0.06248 | 0.08434 | 5.82 | 0.639 |
| FakeTorino | A10 | 0.03199 | 0.03162 | 5.82 | 0.622 |
| FakeTorino | L3TM | 0.03239 | 0.03148 | 6.28 | 0.020 |
| FakeKingston | A9 | 0.05757 | 0.08882 | 5.88 | 0.773 |
| FakeKingston | A10 | 0.01722 | 0.01794 | 5.89 | 0.760 |
| FakeKingston | L3TM | 0.01767 | 0.01764 | 6.30 | 0.021 |
| FakeAuckland | A9 | 0.03413 | 0.02886 | 5.80 | 0.215 |
| FakeAuckland | A10 | 0.03329 | 0.02649 | 5.80 | 0.218 |
| FakeAuckland | L3TM | 0.03557 | 0.02390 | 6.29 | 0.013 |

- M1 (A10 puts the measured qubits on better readout: A10/A9 summed measure error <= 0.80 on both Heron devices, <= 1 everywhere): **CONFIRMED** (FakeTorino 0.375, FakeKingston 0.202, FakeAuckland 0.918)
- M2 (the sampled distribution is more faithful: A10/A9 classical infidelity <= 1.00 everywhere and <= 0.95 on both Heron devices): **CONFIRMED** (FakeTorino 0.512, FakeKingston 0.299, FakeAuckland 0.975)
- M3 (A10 at or ahead of L3TM on every device): **CONFIRMED** (FakeTorino 0.987, FakeKingston 0.974, FakeAuckland 0.936)
- M4 (per circuit A10 <= A9 in >= 80% of circuits on both Heron devices): **CONFIRMED** (FakeTorino 0.876, FakeKingston 0.837)
- M5 (median compile time A10 <= 1.2 x A9): **CONFIRMED** (max 1.014)

SUMMARY {"M1": "CONFIRMED", "M2": "CONFIRMED", "M3": "CONFIRMED", "M4": "CONFIRMED", "M5": "CONFIRMED"}
