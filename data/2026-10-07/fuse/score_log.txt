# FUSE score

files 6, git_head ['c4f943b'], flags []
P0 PASS: errors 0, off-target 0, inexact 0 (checked 54, expected 54)

| device | circuits | identical | n=16: median c19/release | n=12-14 | n=20 | n=8 |
|---|---|---|---|---|---|---|
| FakeTorino | 60 | 60 | 0.397 | 0.649 | 1.104 | 0.987 |
| FakeKingston | 60 | 59 | 0.418 | 0.606 | 0.933 | 0.982 |
| FakeAuckland | 60 | 60 | 0.485 | 0.530 | 1.035 | 0.964 |
| FakeHanoiV2 | 60 | 60 | 0.460 | 0.530 | 1.035 | 0.950 |
| FakeBrussels | 60 | 60 | 0.357 | 0.607 | 1.004 | 0.981 |
| FakeOsaka | 60 | 60 | 0.359 | 0.646 | 0.991 | 0.948 |

| | prediction | verdict |
|---|---|---|
| F1 | c19 returns the release's circuit | **AMBIGUOUS** |
| F2 | n=16: median per-circuit time ratio <= 0.6 on every device | **CONFIRMED** |
| F3 | n=12-14: median per-circuit time ratio <= 0.9 on every device | **CONFIRMED** |
| F4 | n=20: median per-circuit time ratio <= 1.10 on every device | **AMBIGUOUS** |

Circuits that differ: 1.

Reported without prediction (median c19/release per family at n=16 | n=12-14):
- ring: 0.455 (18) | 0.735 (36)
- brick: 0.350 (18) | 0.578 (36)
- pauli: 0.469 (18) | 0.520 (36)
- qft: 0.410 (18) | 0.531 (36)
- n=8: total compile time release 47.1 s, c19 44.9 s (0.953)
- n=12: total compile time release 363.8 s, c19 185.3 s (0.509)
- n=14: total compile time release 686.6 s, c19 353.1 s (0.514)
- n=16: total compile time release 5982.2 s, c19 2495.1 s (0.417)
- n=20: total compile time release 7.6 s, c19 7.8 s (1.029)
