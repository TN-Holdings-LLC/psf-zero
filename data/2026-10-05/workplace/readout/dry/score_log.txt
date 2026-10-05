# READOUT score (DRY RUN -- not a result)

P0: PASS -- devices 3/3; max state infidelity 7.88e-15 (<= 1e-6); measured qubit = final position of logical 0 in every measured arm: True

| device | arm | measure error of the output qubit | eff. margin | acc 32 shots | acc 4000 shots | 2q | median compile s |
|---|---|---|---|---|---|---|---|
| FakeTorino | C12 | 0.0484 | 0.4681 | 0.9439 | 1.0000 | 114.0 | 0.635 |
| FakeTorino | C12M | 0.0084 | 0.4951 | 0.9596 | 1.0000 | 114.0 | 0.615 |
| FakeTorino | C13M | 0.0084 | 0.4951 | 0.9596 | 1.0000 | 114.0 | 0.627 |
| FakeTorino | L3TM | 0.0085 | 0.4889 | 0.9583 | 1.0000 | 109.2 | 0.029 |
| FakeKingston | C12 | 0.0117 | 0.5550 | 0.9736 | 1.0000 | 109.5 | 0.671 |
| FakeKingston | C12M | 0.0101 | 0.5575 | 0.9738 | 1.0000 | 109.5 | 0.681 |
| FakeKingston | C13M | 0.0093 | 0.5583 | 0.9738 | 1.0000 | 109.5 | 0.675 |
| FakeKingston | L3TM | 0.0083 | 0.5590 | 0.9737 | 1.0000 | 109.5 | 0.029 |
| FakeAuckland | C12 | 0.0076 | 0.4124 | 0.9259 | 0.9969 | 109.4 | 0.441 |
| FakeAuckland | C12M | 0.0066 | 0.4140 | 0.9260 | 0.9969 | 109.5 | 0.421 |
| FakeAuckland | C13M | 0.0064 | 0.4141 | 0.9260 | 0.9969 | 109.4 | 0.459 |
| FakeAuckland | L3TM | 0.0070 | 0.4057 | 0.9220 | 0.9969 | 109.5 | 0.026 |

- R1 (compiling with the measurement moves the output to a better-readout qubit, FakeTorino: C12M/C12 measure error <= 0.70): **CONFIRMED** (0.173)
- R2 (c13's readout term never worse, somewhere better: C13M - C12M measure error <= 0 on every device, < 0 on one): **CONFIRMED** (FakeTorino +0.00000, FakeKingston -0.00072, FakeAuckland -0.00014)
- R3 (effective margin C13M - C12 >= +0.01 on FakeTorino): **CONFIRMED** (+0.0270)
- R4 (C13M level with or above L3TM in effective margin, every device, >= -0.01): **CONFIRMED** (FakeTorino +0.0062, FakeKingston -0.0007, FakeAuckland +0.0084)
- R5 (without measurements c13 equals c12, instruction by instruction): **CONFIRMED** (1.0000)
- R6 (median compile time C13M <= 1.2 x C12M on every device): **CONFIRMED** (max ratio 1.090)
- R7 (32-shot accuracy C13M >= C12 on FakeTorino): **CONFIRMED** (+0.0157)

SUMMARY {"R1": "CONFIRMED", "R2": "CONFIRMED", "R3": "CONFIRMED", "R4": "CONFIRMED", "R5": "CONFIRMED", "R6": "CONFIRMED", "R7": "CONFIRMED"}
