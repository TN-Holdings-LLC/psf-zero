# SKIP score

files 6, git_head ['2420fa7'], flags []
P0 PASS: errors 0, off-target 0, inexact 0 (checked 54, expected 54)

| device | circuits | identical | above 16: median c17/release | up to 16: median c17/release |
|---|---|---|---|---|
| FakeTorino | 48 | 48 | 0.242 | 1.017 |
| FakeKingston | 48 | 48 | 0.238 | 1.028 |
| FakeAuckland | 51 | 51 | 0.212 | 1.006 |
| FakeHanoiV2 | 51 | 51 | 0.233 | 1.018 |
| FakeBrussels | 48 | 48 | 0.213 | 1.003 |
| FakeOsaka | 48 | 48 | 0.277 | 0.996 |

| | prediction | verdict |
|---|---|---|
| K1 | c17 returns the release's circuit | **CONFIRMED** |
| K2 | level 3 and the floor skipped once per circuit above 16 qubits, never up to 16 | **CONFIRMED** |
| K3 | above 16: median per-circuit time ratio <= 0.5 on every device | **CONFIRMED** |
| K4 | up to 16: median per-circuit time ratio <= 1.10 on every device | **CONFIRMED** |

Reported without prediction (median c17/release per family, above 16 | up to 16):
- ring: 0.267 (36) | 1.025 (36)
- brick: 0.313 (36) | 1.009 (36)
- pauli: 0.174 (36) | 1.016 (36)
- qft: 0.217 (36) | 0.998 (36)
- fullT: 0.013 (6) | -
- total compile time: release 1339.3 s, c17 1274.5 s (0.952)
- resynthesis skipped on 150 circuits
