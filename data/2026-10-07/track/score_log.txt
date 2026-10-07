# TRACK score

files 6, git_head ['6dbadfa'], flags []
P0 PASS: errors 0, off-target 0, inputs with a wide instruction 0, inexact 0 (checked 54, expected 54)

| device | circuits | identical | n=16: median c22/c19 | n=12-14 | n=20 | n=8 |
|---|---|---|---|---|---|---|
| FakeTorino | 60 | 60 | 0.395 | 0.656 | 1.035 | 0.782 |
| FakeKingston | 60 | 60 | 0.396 | 0.634 | 1.006 | 0.800 |
| FakeAuckland | 60 | 60 | 0.499 | 0.642 | 1.027 | 0.785 |
| FakeHanoiV2 | 60 | 60 | 0.506 | 0.675 | 0.910 | 0.819 |
| FakeBrussels | 60 | 60 | 0.365 | 0.592 | 0.964 | 0.756 |
| FakeOsaka | 60 | 60 | 0.386 | 0.601 | 0.995 | 0.774 |

| | prediction | verdict |
|---|---|---|
| T1 | c22 returns c19's circuit | **CONFIRMED** |
| T2 | n=16: median per-circuit time ratio <= 0.6 on every device | **CONFIRMED** |
| T3 | n=12-14: median per-circuit time ratio <= 0.85 on every device | **CONFIRMED** |
| T4 | n=20: median per-circuit time ratio <= 1.15 on every device | **CONFIRMED** |

Circuits that differ: 0.

Reported without prediction (median c22/c19 per family at n=16 | n=12-14):
- ring: 0.402 (18) | 0.678 (36)
- brick: 0.379 (18) | 0.644 (36)
- pauli: 0.438 (18) | 0.542 (36)
- qft: 0.407 (18) | 0.627 (36)
- n=8: total compile time c19 47.2 s, c22 35.7 s (0.757)
- n=12: total compile time c19 132.2 s, c22 81.8 s (0.619)
- n=14: total compile time c19 339.1 s, c22 166.7 s (0.492)
- n=16: total compile time c19 1519.9 s, c22 644.1 s (0.424)
- n=20: total compile time c19 9.3 s, c22 8.7 s (0.940)
