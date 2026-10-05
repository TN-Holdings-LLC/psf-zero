# MODEL-RO2 score (DRY RUN -- not a result)

P0: PASS -- devices 3/3, circuits per device [12, 12, 12] (expected 12); max state infidelity 3.77e-15 (<= 1e-6; total variation, reported only: 4.67e-15); clbit j measures logical j's final qubit: True

| device | arm | classical infidelity | summed measure error | 2q | median compile s |
|---|---|---|---|---|---|
| FakeTorino | A9 | 0.01075 | 0.13442 | 8.67 | 0.772 |
| FakeTorino | A10 | 0.00580 | 0.04791 | 8.83 | 0.721 |
| FakeTorino | L3TM | 0.00686 | 0.05031 | 9.17 | 0.019 |
| FakeKingston | A9 | 0.01077 | 0.13161 | 9.00 | 1.140 |
| FakeKingston | A10 | 0.00194 | 0.02533 | 9.00 | 1.082 |
| FakeKingston | L3TM | 0.00203 | 0.02463 | 9.67 | 0.022 |
| FakeAuckland | A9 | 0.00664 | 0.03514 | 8.42 | 0.281 |
| FakeAuckland | A10 | 0.00641 | 0.03315 | 8.58 | 0.281 |
| FakeAuckland | L3TM | 0.00667 | 0.03122 | 8.92 | 0.012 |

- M1 (A10 puts the measured qubits on better readout: A10/A9 summed measure error <= 0.80 on both Heron devices, <= 1 everywhere): **CONFIRMED** (FakeTorino 0.356, FakeKingston 0.192, FakeAuckland 0.943)
- M2 (the sampled distribution is more faithful: A10/A9 classical infidelity <= 1.00 everywhere and <= 0.95 on both Heron devices): **CONFIRMED** (FakeTorino 0.540, FakeKingston 0.180, FakeAuckland 0.965)
- M3 (A10 at or ahead of L3TM on every device): **CONFIRMED** (FakeTorino 0.845, FakeKingston 0.957, FakeAuckland 0.960)
- M4 (per circuit A10 <= A9 in >= 80% of circuits on both Heron devices): **CONFIRMED** (FakeTorino 0.833, FakeKingston 0.917)
- M5 (median compile time A10 <= 1.2 x A9): **CONFIRMED** (max 1.000)

SUMMARY {"M1": "CONFIRMED", "M2": "CONFIRMED", "M3": "CONFIRMED", "M4": "CONFIRMED", "M5": "CONFIRMED"}
