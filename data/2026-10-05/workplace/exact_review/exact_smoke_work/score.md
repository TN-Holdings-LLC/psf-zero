# exact score (SMOKE -- not a result)

P0: PASS -- files 36 of 36 (X), 9 of 9 (Y); missing []; compile errors 0

| device | cell | RPSF | R41 | C12 | L3T | A8 | A9 |
|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | X1 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| FakeAuckland (cx) | X2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeAuckland (cx) | X3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeAuckland (cx) | X4 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeAuckland (cx) | X5 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| FakeHanoiV2 (cx) | X1 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| FakeHanoiV2 (cx) | X2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeHanoiV2 (cx) | X3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeHanoiV2 (cx) | X4 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeHanoiV2 (cx) | X5 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| FakeAlgiers (cx) | X1 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| FakeAlgiers (cx) | X2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeAlgiers (cx) | X3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeAlgiers (cx) | X4 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeAlgiers (cx) | X5 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| FakeGeneva (cx) | X1 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| FakeGeneva (cx) | X2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeGeneva (cx) | X3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeGeneva (cx) | X4 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeGeneva (cx) | X5 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| FakeTorino | X1 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| FakeTorino | X2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeTorino | X3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeTorino | X4 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeTorino | X5 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| FakeKingston | X1 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| FakeKingston | X2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeKingston | X3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeKingston | X4 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| FakeKingston | X5 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |

## Predictions

- E1 (C12 never wrong, part X): **CONFIRMED** (0 wrong)
- E2 (A9 never wrong, part X): **CONFIRMED** (0 wrong)
- E3 (the defect reproduces: R41 wrong on W2 on at least 3 of 4 cx devices): **REFUTED** ({'FakeAuckland': 0, 'FakeHanoiV2': 0, 'FakeAlgiers': 0, 'FakeGeneva': 0})
- E4 (cz devices unaffected: R41 never wrong on FakeTorino, FakeKingston): **CONFIRMED** (0 wrong)
- E5 (the guarded call RPSF never wrong): **CONFIRMED** (0 wrong)
- E6 (C12 identical to R41 on >= 99.5% of HOLD6's F circuits on every device): **CONFIRMED** (Auckland 1.0000, Torino 1.0000, Kingston 1.0000, HanoiV2 1.0000, Algiers 1.0000, Geneva 1.0000, Fez 1.0000, Marrakesh 1.0000, Aachen 1.0000)
- E7 (A9 identical to A8 on >= 99% of the sampled F circuits on every device): **CONFIRMED** (Auckland 1.0000, Torino 1.0000, Kingston 1.0000, HanoiV2 1.0000, Algiers 1.0000, Geneva 1.0000, Fez 1.0000, Marrakesh 1.0000, Aachen 1.0000)
- E8 (C12 exact on every HOLD6 F circuit): **CONFIRMED** (0 wrong)
- E9 (median compile time C12 <= 1.2 x R41 on the F circuits): **AMBIGUOUS** (0.313 s vs 0.260 s)

Reported: A8 wrong by device {'FakeAuckland': (0, 16), 'FakeHanoiV2': (0, 16), 'FakeAlgiers': (0, 16), 'FakeGeneva': (0, 16), 'FakeTorino': (0, 16), 'FakeKingston': (0, 16)}
Reported: L3T wrong by device {'FakeAuckland': (0, 16), 'FakeHanoiV2': (0, 16), 'FakeAlgiers': (0, 16), 'FakeGeneva': (0, 16), 'FakeTorino': (0, 16), 'FakeKingston': (0, 16)}
Reported: C12 EXACT_STATS by device (part X) {'FakeAuckland': {'checked': 64, 'refused_resynthesis': 8, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeHanoiV2': {'checked': 64, 'refused_resynthesis': 8, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeAlgiers': {'checked': 64, 'refused_resynthesis': 8, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeGeneva': {'checked': 64, 'refused_resynthesis': 8, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeTorino': {'checked': 64, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeKingston': {'checked': 64, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}}
Reported: A9 L3T_CHECK_STATS by device (part X) {'FakeAuckland': {'accepted': 16, 'refused': 0, 'unavailable': 0}, 'FakeHanoiV2': {'accepted': 16, 'refused': 0, 'unavailable': 0}, 'FakeAlgiers': {'accepted': 16, 'refused': 0, 'unavailable': 0}, 'FakeGeneva': {'accepted': 16, 'refused': 0, 'unavailable': 0}, 'FakeTorino': {'accepted': 16, 'refused': 0, 'unavailable': 0}, 'FakeKingston': {'accepted': 16, 'refused': 0, 'unavailable': 0}}
Reported: C12 EXACT_STATS by device (part Y) {'FakeAachen': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeAlgiers': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeAuckland': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeFez': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeGeneva': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeHanoiV2': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeKingston': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeMarrakesh': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}, 'FakeTorino': {'checked': 78, 'refused_resynthesis': 0, 'refused_floor': 0, 'refused_level3': 0, 'not_checkable': 0}}
Reported: A9 exact on the sampled F circuits: 0 wrong
