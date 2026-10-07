# TRACK score (SMOKE: not counted)

files 6, git_head ['9afba52'], flags []
P0 PASS: errors 0, off-target 0, inputs with a wide instruction 0, inexact 0 (checked 18, expected 18)

| device | circuits | identical | n=16: median c22/c19 | n=12-14 | n=20 | n=8 |
|---|---|---|---|---|---|---|
| FakeTorino | 20 | 20 | 0.339 | 0.592 | 1.108 | 0.758 |
| FakeKingston | 20 | 20 | 0.375 | 0.572 | 1.012 | 0.774 |
| FakeAuckland | 20 | 20 | 0.539 | 0.636 | 1.053 | 0.761 |
| FakeHanoiV2 | 20 | 20 | 0.610 | 0.686 | 0.872 | 0.794 |
| FakeBrussels | 20 | 20 | 0.381 | 0.628 | 0.976 | 0.626 |
| FakeOsaka | 20 | 20 | 0.374 | 0.629 | 0.975 | 0.733 |

| | prediction | verdict |
|---|---|---|
| T1 | c22 returns c19's circuit | **CONFIRMED** |
| T2 | n=16: median per-circuit time ratio <= 0.6 on every device | **AMBIGUOUS** |
| T3 | n=12-14: median per-circuit time ratio <= 0.85 on every device | **CONFIRMED** |
| T4 | n=20: median per-circuit time ratio <= 1.15 on every device | **CONFIRMED** |

Circuits that differ: 0.

Reported without prediction (median c22/c19 per family at n=16 | n=12-14):
- ring: 0.408 (6) | 0.769 (12)
- brick: 0.338 (6) | 0.658 (12)
- pauli: 0.496 (6) | 0.544 (12)
- qft: 0.290 (6) | 0.599 (12)
- n=8: total compile time c19 12.7 s, c22 9.2 s (0.724)
- n=12: total compile time c19 70.9 s, c22 32.5 s (0.458)
- n=14: total compile time c19 68.1 s, c22 36.1 s (0.530)
- n=16: total compile time c19 518.0 s, c22 228.2 s (0.441)
- n=20: total compile time c19 2.6 s, c22 2.6 s (0.966)
