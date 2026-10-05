# READOUT score

P0: PASS -- devices 3/3; max state infidelity 1.12e-09 (<= 1e-6); measured qubit = final position of logical 0 in every measured arm: True

| device | arm | measure error of the output qubit | eff. margin | acc 32 shots | acc 4000 shots | 2q | median compile s |
|---|---|---|---|---|---|---|---|
| FakeTorino | C12 | 0.0484 | 0.4908 | 0.9325 | 0.9623 | 114.0 | 0.640 |
| FakeTorino | C12M | 0.0084 | 0.5252 | 0.9421 | 0.9625 | 114.0 | 0.639 |
| FakeTorino | C13M | 0.0084 | 0.5252 | 0.9421 | 0.9625 | 114.0 | 0.638 |
| FakeTorino | L3TM | 0.0086 | 0.5186 | 0.9414 | 0.9625 | 109.2 | 0.029 |
| FakeKingston | C12 | 0.0117 | 0.5992 | 0.9483 | 0.9625 | 109.5 | 0.675 |
| FakeKingston | C12M | 0.0101 | 0.6021 | 0.9485 | 0.9625 | 109.5 | 0.676 |
| FakeKingston | C13M | 0.0084 | 0.6039 | 0.9487 | 0.9625 | 109.5 | 0.662 |
| FakeKingston | L3TM | 0.0083 | 0.6034 | 0.9486 | 0.9625 | 109.5 | 0.031 |
| FakeAuckland | C12 | 0.0075 | 0.4254 | 0.9137 | 0.9625 | 109.4 | 0.427 |
| FakeAuckland | C12M | 0.0066 | 0.4270 | 0.9141 | 0.9625 | 109.5 | 0.423 |
| FakeAuckland | C13M | 0.0066 | 0.4270 | 0.9141 | 0.9625 | 109.5 | 0.422 |
| FakeAuckland | L3TM | 0.0070 | 0.4176 | 0.9120 | 0.9625 | 109.5 | 0.024 |

- R1 (compiling with the measurement moves the output to a better-readout qubit, FakeTorino: C12M/C12 measure error <= 0.70): **CONFIRMED** (0.173)
- R2 (c13's readout term never worse, somewhere better: C13M - C12M measure error <= 0 on every device, < 0 on one): **CONFIRMED** (FakeTorino +0.00000, FakeKingston -0.00170, FakeAuckland -0.00002)
- R3 (effective margin C13M - C12 >= +0.01 on FakeTorino): **CONFIRMED** (+0.0344)
- R4 (C13M level with or above L3TM in effective margin, every device, >= -0.01): **CONFIRMED** (FakeTorino +0.0066, FakeKingston +0.0005, FakeAuckland +0.0094)
- R5 (without measurements c13 equals c12, instruction by instruction): **CONFIRMED** (1.0000)
- R6 (median compile time C13M <= 1.2 x C12M on every device): **CONFIRMED** (max ratio 0.998)
- R7 (32-shot accuracy C13M >= C12 on FakeTorino): **CONFIRMED** (+0.0096)

SUMMARY {"R1": "CONFIRMED", "R2": "CONFIRMED", "R3": "CONFIRMED", "R4": "CONFIRMED", "R5": "CONFIRMED", "R6": "CONFIRMED", "R7": "CONFIRMED"}
