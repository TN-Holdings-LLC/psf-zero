# KRAUS score (SMOKE)

P0: PASS -- files 54 of 54; errors 0; inexact candidates 0; inconsistent choices 0; counts ok True

| device | HYB | KRA | PAU | BEST | KRA/HYB | KRA/L3 | KRA=BEST | HYB=BEST | changed | KRA better |
|---|---|---|---|---|---|---|---|---|---|---|
| FakeAuckland | 0.26756 | 0.26722 | 0.26767 | 0.26720 | 0.9987 | 0.9795 | 17/19 | 16/19 | 3 | 2 |
| FakeTorino | 0.14496 | 0.14496 | 0.14529 | 0.14496 | 1.0000 | 0.9738 | 17/19 | 17/19 | 2 | 1 |
| FakeKingston | 0.07135 | 0.07135 | 0.07139 | 0.07135 | 1.0000 | 0.9883 | 18/19 | 18/19 | 2 | 1 |
| FakeHanoiV2 | 0.21718 | 0.21702 | 0.21784 | 0.21701 | 0.9993 | 0.9835 | 18/19 | 18/19 | 2 | 1 |
| FakeAlgiers | 0.24979 | 0.24858 | 0.24975 | 0.24858 | 0.9951 | 0.9714 | 19/19 | 15/19 | 4 | 4 |
| FakeGeneva | 0.21572 | 0.21545 | 0.21572 | 0.21545 | 0.9988 | 0.9394 | 19/19 | 18/19 | 1 | 1 |
| FakeFez | 0.13550 | 0.13546 | 0.13573 | 0.13546 | 0.9998 | 0.9924 | 19/19 | 17/19 | 2 | 2 |
| FakeMarrakesh | 0.08717 | 0.08701 | 0.08743 | 0.08701 | 0.9982 | 0.9517 | 18/19 | 18/19 | 2 | 1 |
| FakeAachen | 0.06229 | 0.06225 | 0.06230 | 0.06225 | 0.9995 | 0.9843 | 19/19 | 15/19 | 4 | 4 |

## Predictions

- K1 (kraus chooses at least as well as hybrid overall (KRA/HYB <= 1.000 on >= 8 of 9 devices; refuted > 1.002 on any)): **CONFIRMED** -- {"FakeAuckland": 0.9987082417267007, "FakeTorino": 1.000000032113635, "FakeKingston": 0.9999999460651477, "FakeHanoiV2": 0.9992577841169443, "FakeAlgiers": 0.9951357255824476, "FakeGeneva": 0.9987810765784787, "FakeFez": 0.9997582192995764, "FakeMarrakesh": 0.998222088100248, "FakeAachen": 0.9994544208720249}
- K2 (the H4 case is repaired (FakeAlgiers F5 n = 4: KRA/HYB <= 0.96; refuted >= 1.00)): **CONFIRMED** -- {"ratio": 0.9372262290551565, "circuits": 1}
- K3 (kraus is closer to the measured best (gap KRA <= 0.5 x gap HYB on >= 7 of 9 devices; refuted gap KRA > gap HYB on >= 3)): **CONFIRMED** -- {"FakeAuckland": [7.639376497303729e-05, 0.0013699216458922248], "FakeTorino": [5.167493744018259e-08, 1.9561301689563493e-08], "FakeKingston": [6.598589341599848e-08, 1.1992075199529495e-07], "FakeHanoiV2": [5.089139016556388e-05, 0.0007936963672714459], "FakeAlgiers": [0.0, 0.004888051240151681], "FakeGeneva": [0.0, 0.0012204110090840992], "FakeFez": [0.0, 0.00024183917246811681], "FakeMarrakesh": [1.9746273061116426e-07, 0.0017812763148394861], "FakeAachen": [0.0, 0.0005458769470438884]}
- K4 (where the choice changes, kraus is better more often (>= 60% on every device with >= 20 changes; refuted < 50% on any)): **AMBIGUOUS** -- {}
- K5 (no family loses (KRA/HYB <= 1.01 in every family-device cell; refuted > 1.03 in any)): **CONFIRMED** -- {}
- K6 (kraus stays ahead of Qiskit level 3 (KRA/L3 <= 1.00 on 9 of 9; refuted > 1.02 on any)): **CONFIRMED** -- {"FakeAuckland": 0.9795401697737108, "FakeTorino": 0.9737642809129006, "FakeKingston": 0.9883056974391358, "FakeHanoiV2": 0.9834678107297901, "FakeAlgiers": 0.9714305512690625, "FakeGeneva": 0.9394378119008858, "FakeFez": 0.9924426036287706, "FakeMarrakesh": 0.9517305392387968, "FakeAachen": 0.9843098763990924}
- K7 (it costs little (median compile time of K, as the release's call plus kraus_cost's extra time over hybrid_cost on the same candidates, / R <= 1.15 on every device; refuted > 1.50 on any)): **CONFIRMED** -- {"FakeAuckland": 1.0678221552373774, "FakeTorino": 1.0337078651685394, "FakeKingston": 1.0282819755170958, "FakeHanoiV2": 1.0793991416309012, "FakeAlgiers": 1.0286738351254479, "FakeGeneva": 1.0422535211267605, "FakeFez": 1.0083449235048678, "FakeMarrakesh": 1.021808510638298, "FakeAachen": 1.0119104335397808}

Reported: family-device KRA/HYB cells: {"FakeAuckland/F1": 0.9968, "FakeAuckland/F2": 1.0, "FakeAuckland/F3": 1.0, "FakeAuckland/F4": 1.0, "FakeAuckland/F5": 1.0, "FakeAuckland/F6": 1.0, "FakeTorino/F1": 1.0, "FakeTorino/F2": 1.0, "FakeTorino/F3": 1.0, "FakeTorino/F4": 1.0, "FakeTorino/F5": 1.0, "FakeTorino/F6": 1.0, "FakeKingston/F1": 1.0, "FakeKingston/F2": 1.0, "FakeKingston/F3": 1.0, "FakeKingston/F4": 1.0, "FakeKingston/F5": 1.0, "FakeKingston/F6": 1.0, "FakeHanoiV2/F1": 1.0001, "FakeHanoiV2/F2": 1.0, "FakeHanoiV2/F3": 1.0, "FakeHanoiV2/F4": 1.0, "FakeHanoiV2/F5": 1.0, "FakeHanoiV2/F6": 0.9932, "FakeAlgiers/F1": 0.9999, "FakeAlgiers/F2": 1.0, "FakeAlgiers/F3": 0.9724, "FakeAlgiers/F4": 1.0, "FakeAlgiers/F5": 0.9877, "FakeAlgiers/F6": 0.9978, "FakeGeneva/F1": 0.997, "FakeGeneva/F2": 1.0, "FakeGeneva/F3": 1.0, "FakeGeneva/F4": 1.0, "FakeGeneva/F5": 1.0, "FakeGeneva/F6": 1.0, "FakeFez/F1": 1.0, "FakeFez/F2": 1.0, "FakeFez/F3": 1.0, "FakeFez/F4": 0.9984, "FakeFez/F5": 1.0, "FakeFez/F6": 1.0, "FakeMarrakesh/F1": 1.0, "FakeMarrakesh/F2": 1.0, "FakeMarrakesh/F3": 0.9907, "FakeMarrakesh/F4": 1.0, "FakeMarrakesh/F5": 1.0, "FakeMarrakesh/F6": 1.0, "FakeAachen/F1": 1.0, "FakeAachen/F2": 1.0, "FakeAachen/F3": 1.0, "FakeAachen/F4": 1.0, "FakeAachen/F5": 1.0, "FakeAachen/F6": 0.995}
Reported: measured K / R median compile time (order alternates; a second call runs warm): {"FakeAuckland": 0.669, "FakeTorino": 0.807, "FakeKingston": 0.726, "FakeHanoiV2": 0.652, "FakeAlgiers": 0.676, "FakeGeneva": 0.6, "FakeFez": 0.593, "FakeMarrakesh": 0.851, "FakeAachen": 0.813}
Reported: candidates too wide to simulate are excluded from every mean: {"FakeAuckland": 0, "FakeTorino": 0, "FakeKingston": 0, "FakeHanoiV2": 0, "FakeAlgiers": 0, "FakeGeneva": 0, "FakeFez": 0, "FakeMarrakesh": 0, "FakeAachen": 0}
