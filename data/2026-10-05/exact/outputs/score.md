# exact score

P0: PASS -- files 36 of 36 (X), 9 of 9 (Y); missing []; compile errors 0

| device | cell | RPSF | R41 | C12 | L3T | A8 | A9 |
|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | X1 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| FakeAuckland (cx) | X2 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| FakeAuckland (cx) | X3 | 0/40 | 23/40 (3.4e-01) | 0/40 | 28/40 (3.4e-01) | 27/40 (3.4e-01) | 0/40 |
| FakeAuckland (cx) | X4 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeAuckland (cx) | X5 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeHanoiV2 (cx) | X1 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| FakeHanoiV2 (cx) | X2 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| FakeHanoiV2 (cx) | X3 | 0/40 | 25/40 (3.4e-01) | 0/40 | 28/40 (3.4e-01) | 27/40 (3.4e-01) | 0/40 |
| FakeHanoiV2 (cx) | X4 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeHanoiV2 (cx) | X5 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeAlgiers (cx) | X1 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| FakeAlgiers (cx) | X2 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| FakeAlgiers (cx) | X3 | 0/40 | 26/40 (3.4e-01) | 0/40 | 28/40 (3.4e-01) | 27/40 (3.4e-01) | 0/40 |
| FakeAlgiers (cx) | X4 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeAlgiers (cx) | X5 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeGeneva (cx) | X1 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| FakeGeneva (cx) | X2 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| FakeGeneva (cx) | X3 | 0/40 | 27/40 (3.4e-01) | 0/40 | 28/40 (3.4e-01) | 27/40 (3.4e-01) | 0/40 |
| FakeGeneva (cx) | X4 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeGeneva (cx) | X5 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeTorino | X1 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| FakeTorino | X2 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| FakeTorino | X3 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| FakeTorino | X4 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeTorino | X5 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeKingston | X1 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| FakeKingston | X2 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| FakeKingston | X3 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| FakeKingston | X4 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| FakeKingston | X5 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |

## Predictions

- E1 (C12 never wrong, part X): **CONFIRMED** (0 wrong)
- E2 (A9 never wrong, part X): **CONFIRMED** (0 wrong)
- E3 (the defect reproduces: R41 wrong on W2 on at least 3 of 4 cx devices): **CONFIRMED** ({'FakeAuckland': 23, 'FakeHanoiV2': 25, 'FakeAlgiers': 26, 'FakeGeneva': 27})
- E4 (cz devices unaffected: R41 never wrong on FakeTorino, FakeKingston): **CONFIRMED** (0 wrong)
- E5 (the guarded call RPSF never wrong): **CONFIRMED** (0 wrong)
- E6 (C12 identical to R41 on >= 99.5% of HOLD6's F circuits on every device): **CONFIRMED** (Auckland 1.0000, Torino 1.0000, Kingston 1.0000, HanoiV2 1.0000, Algiers 1.0000, Geneva 1.0000, Fez 1.0000, Marrakesh 1.0000, Aachen 1.0000)
- E7 (A9 identical to A8 on >= 99% of the sampled F circuits on every device): **CONFIRMED** (Auckland 1.0000, Torino 1.0000, Kingston 1.0000, HanoiV2 1.0000, Algiers 1.0000, Geneva 1.0000, Fez 1.0000, Marrakesh 1.0000, Aachen 1.0000)
- E8 (C12 exact on every HOLD6 F circuit): **CONFIRMED** (0 wrong)
- E9 (median compile time C12 <= 1.2 x R41 on the F circuits): **AMBIGUOUS** (0.194 s vs 0.154 s)

Reported: A8 wrong by device {'FakeAuckland': (27, 128), 'FakeHanoiV2': (27, 128), 'FakeAlgiers': (27, 128), 'FakeGeneva': (27, 128), 'FakeTorino': (0, 128), 'FakeKingston': (0, 128)}
Reported: L3T wrong by device {'FakeAuckland': (28, 128), 'FakeHanoiV2': (28, 128), 'FakeAlgiers': (28, 128), 'FakeGeneva': (28, 128), 'FakeTorino': (0, 128), 'FakeKingston': (0, 128)}
Reported: C12 EXACT_STATS by device (part X) {'FakeAuckland': {'checked': 512, 'refused_resynthesis': 128, 'refused_floor': 0, 'refused_level3': 28, 'not_checkable': 0}, 'FakeHanoiV2': {'checked': 512, 'refused_resynthesis': 128, 'refused_floor': 0, 'refused_level3': 28, 'not_checkable': 0}, 'FakeAlgiers': {'checked': 512, 'refused_resynthesis': 128, 'refused_floor': 0, 'refused_level3': 28, 'not_checkable': 0}, 'FakeGeneva': {'checked': 512, 'refused_resynthesis': 128, 'refused_floor': 0, 'refused_level3': 28, 'not_checkable': 0}, 'FakeTorino': {'checked': 512, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeKingston': {'checked': 512, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}}
Reported: A9 L3T_CHECK_STATS by device (part X) {'FakeAuckland': {'accepted': 100, 'refused': 28, 'unavailable': 0}, 'FakeHanoiV2': {'accepted': 100, 'refused': 28, 'unavailable': 0}, 'FakeAlgiers': {'accepted': 100, 'refused': 28, 'unavailable': 0}, 'FakeGeneva': {'accepted': 100, 'refused': 28, 'unavailable': 0}, 'FakeTorino': {'accepted': 128, 'refused': 0, 'unavailable': 0}, 'FakeKingston': {'accepted': 128, 'refused': 0, 'unavailable': 0}}
Reported: C12 EXACT_STATS by device (part Y) {'FakeAachen': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeAlgiers': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeAuckland': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeFez': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeGeneva': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeHanoiV2': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeKingston': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeMarrakesh': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeTorino': {'checked': 6175, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}}
Reported: A9 exact on the sampled F circuits: 0 wrong
