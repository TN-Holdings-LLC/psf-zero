# MODEL-RO score (DRY RUN -- not a result)

P0: PASS -- devices 3/3, circuits per device [6, 6, 6] (expected 6); max noiseless total variation 7.99e-16 (<= 1e-6); clbit j measures logical j's final qubit: True

| device | arm | classical infidelity | summed measure error | 2q | median compile s |
|---|---|---|---|---|---|
| FakeTorino | A9 | 0.02577 | 0.13224 | 3.00 | 0.599 |
| FakeTorino | A10 | 0.00583 | 0.04606 | 3.00 | 0.601 |
| FakeTorino | L3TM | 0.00597 | 0.04606 | 3.00 | 0.016 |
| FakeKingston | A9 | 0.03781 | 0.17159 | 3.00 | 0.754 |
| FakeKingston | A10 | 0.00220 | 0.02250 | 3.00 | 0.740 |
| FakeKingston | L3TM | 0.00219 | 0.02250 | 3.00 | 0.020 |
| FakeAuckland | A9 | 0.00516 | 0.03573 | 3.00 | 0.196 |
| FakeAuckland | A10 | 0.00465 | 0.03053 | 3.00 | 0.184 |
| FakeAuckland | L3TM | 0.00449 | 0.03060 | 3.00 | 0.011 |

- M1 (A10 puts the measured qubits on better readout: A10/A9 summed measure error <= 0.80 on both Heron devices, <= 1 everywhere): **CONFIRMED** (FakeTorino 0.348, FakeKingston 0.131, FakeAuckland 0.854)
- M2 (the sampled distribution is more faithful: A10/A9 classical infidelity <= 1.00 everywhere and <= 0.95 on both Heron devices): **CONFIRMED** (FakeTorino 0.226, FakeKingston 0.058, FakeAuckland 0.902)
- M3 (A10 at or ahead of L3TM on every device): **AMBIGUOUS** (FakeTorino 0.976, FakeKingston 1.005, FakeAuckland 1.036)
- M4 (per circuit A10 <= A9 in >= 80% of circuits on both Heron devices): **CONFIRMED** (FakeTorino 0.833, FakeKingston 1.000)
- M5 (median compile time A10 <= 1.2 x A9): **CONFIRMED** (max 1.004)

SUMMARY {"M1": "CONFIRMED", "M2": "CONFIRMED", "M3": "AMBIGUOUS", "M4": "CONFIRMED", "M5": "CONFIRMED"}
