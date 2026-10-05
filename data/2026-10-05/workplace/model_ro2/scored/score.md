# MODEL-RO2 score

P0: PASS -- devices 3/3, circuits per device [200, 200, 200] (expected 200); max state infidelity 4.00e-15 (<= 1e-6; total variation, reported only: 1.21e-13); clbit j measures logical j's final qubit: True

| device | arm | classical infidelity | summed measure error | 2q | median compile s |
|---|---|---|---|---|---|
| FakeTorino | A9 | 0.01335 | 0.12794 | 9.69 | 0.856 |
| FakeTorino | A10 | 0.00466 | 0.04956 | 9.70 | 0.827 |
| FakeTorino | L3TM | 0.00477 | 0.05004 | 9.73 | 0.020 |
| FakeKingston | A9 | 0.01558 | 0.14737 | 10.04 | 1.116 |
| FakeKingston | A10 | 0.00168 | 0.02656 | 10.04 | 1.118 |
| FakeKingston | L3TM | 0.00174 | 0.02457 | 10.21 | 0.020 |
| FakeAuckland | A9 | 0.00585 | 0.03546 | 9.54 | 0.318 |
| FakeAuckland | A10 | 0.00571 | 0.03364 | 9.54 | 0.314 |
| FakeAuckland | L3TM | 0.00656 | 0.03173 | 9.74 | 0.013 |

- M1 (A10 puts the measured qubits on better readout: A10/A9 summed measure error <= 0.80 on both Heron devices, <= 1 everywhere): **CONFIRMED** (FakeTorino 0.387, FakeKingston 0.180, FakeAuckland 0.949)
- M2 (the sampled distribution is more faithful: A10/A9 classical infidelity <= 1.00 everywhere and <= 0.95 on both Heron devices): **CONFIRMED** (FakeTorino 0.349, FakeKingston 0.108, FakeAuckland 0.977)
- M3 (A10 at or ahead of L3TM on every device): **CONFIRMED** (FakeTorino 0.975, FakeKingston 0.966, FakeAuckland 0.870)
- M4 (per circuit A10 <= A9 in >= 80% of circuits on both Heron devices): **CONFIRMED** (FakeTorino 0.935, FakeKingston 0.975)
- M5 (median compile time A10 <= 1.2 x A9): **CONFIRMED** (max 1.002)

SUMMARY {"M1": "CONFIRMED", "M2": "CONFIRMED", "M3": "CONFIRMED", "M4": "CONFIRMED", "M5": "CONFIRMED"}
