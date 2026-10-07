# DEPTH-R score (DRY RUN -- not a result)

P0: PASS -- deploy files 24/24; max state infidelity 8.22e-15 (<= 1e-06); max |reduced - whole-device| 0.00e+00 over 144 circuits (<= 1e-9); finetune files 4; versions ['2026-10-07.1']; seeds [(2, 2)]; release file as locked in every output True
(recorded, not gated: max |z_compiled_noiseless - z_ideal| 2.49e-14)

## FakeAuckland

| dataset | n | L | ideal acc / margin | RPSF acc / shot acc / margin / flip / 2q | REC acc / shot acc / margin / flip / 2q | L3T acc / shot acc / margin / flip / 2q |
|---|---|---|---|---|---|---|
| BC | 4 | 1 | 0.939 / 0.380 | 0.917 / 0.921 / 0.357 / 0.000 / 7 | 0.917 / 0.917 / 0.356 / 0.000 / 7 | 0.917 / 0.917 / 0.353 / 0.000 / 7 |
| BC | 4 | 4 | 0.965 / 0.653 | 1.000 / 1.000 / 0.602 / 0.000 / 37 | 1.000 / 1.000 / 0.603 / 0.000 / 37 | 1.000 / 1.000 / 0.591 / 0.000 / 37 |
| BC | 4 | 12 | 0.947 / 0.624 | 1.000 / 1.000 / 0.403 / 0.000 / 117 | 1.000 / 1.000 / 0.416 / 0.000 / 117 | 1.000 / 1.000 / 0.379 / 0.000 / 117 |
| BC | 6 | 1 | 0.930 / 0.404 | 0.917 / 0.921 / 0.376 / 0.000 / 13 | 0.917 / 0.921 / 0.376 / 0.000 / 13 | 0.917 / 0.921 / 0.375 / 0.000 / 13 |
| BC | 6 | 4 | 0.939 / 0.615 | 1.000 / 1.000 / 0.491 / 0.000 / 71 | 1.000 / 1.000 / 0.512 / 0.000 / 70 | 1.000 / 1.000 / 0.515 / 0.000 / 70 |
| BC | 6 | 12 | 0.939 / 0.574 | 1.000 / 0.992 / 0.215 / 0.000 / 213 | 1.000 / 0.987 / 0.215 / 0.000 / 213 | 1.000 / 0.987 / 0.215 / 0.000 / 214 |
| D38 | 4 | 1 | 0.944 / 0.524 | 1.000 / 1.000 / 0.628 / 0.000 / 7 | 1.000 / 1.000 / 0.627 / 0.000 / 7 | 1.000 / 1.000 / 0.625 / 0.000 / 7 |
| D38 | 4 | 4 | 0.958 / 0.673 | 1.000 / 1.000 / 0.649 / 0.000 / 37 | 1.000 / 1.000 / 0.668 / 0.000 / 37 | 1.000 / 1.000 / 0.649 / 0.000 / 37 |
| D38 | 4 | 12 | 0.944 / 0.529 | 1.000 / 1.000 / 0.287 / 0.000 / 117 | 1.000 / 1.000 / 0.304 / 0.000 / 117 | 1.000 / 1.000 / 0.281 / 0.000 / 117 |
| D38 | 6 | 1 | 0.931 / 0.492 | 1.000 / 1.000 / 0.536 / 0.000 / 13 | 1.000 / 1.000 / 0.536 / 0.000 / 13 | 1.000 / 1.000 / 0.536 / 0.000 / 13 |
| D38 | 6 | 4 | 0.958 / 0.696 | 1.000 / 1.000 / 0.579 / 0.000 / 71 | 1.000 / 1.000 / 0.593 / 0.000 / 70 | 1.000 / 1.000 / 0.598 / 0.000 / 70 |
| D38 | 6 | 12 | 0.958 / 0.537 | 1.000 / 1.000 / 0.247 / 0.000 / 213 | 1.000 / 1.000 / 0.246 / 0.000 / 213 | 1.000 / 1.000 / 0.252 / 0.000 / 214 |

## FakeTorino

| dataset | n | L | ideal acc / margin | RPSF acc / shot acc / margin / flip / 2q | REC acc / shot acc / margin / flip / 2q | L3T acc / shot acc / margin / flip / 2q |
|---|---|---|---|---|---|---|
| BC | 4 | 1 | 0.939 / 0.380 | 0.917 / 0.921 / 0.358 / 0.000 / 8 | 0.917 / 0.921 / 0.359 / 0.000 / 7 | 0.917 / 0.921 / 0.359 / 0.000 / 7 |
| BC | 4 | 4 | 0.965 / 0.653 | 1.000 / 1.000 / 0.613 / 0.000 / 44 | 1.000 / 1.000 / 0.639 / 0.000 / 37 | 1.000 / 1.000 / 0.639 / 0.000 / 37 |
| BC | 4 | 12 | 0.947 / 0.624 | 1.000 / 1.000 / 0.488 / 0.000 / 140 | 1.000 / 1.000 / 0.550 / 0.000 / 117 | 1.000 / 1.000 / 0.550 / 0.000 / 117 |
| BC | 6 | 1 | 0.930 / 0.404 | 0.917 / 0.917 / 0.375 / 0.000 / 17 | 0.917 / 0.917 / 0.375 / 0.000 / 17 | 0.917 / 0.921 / 0.375 / 0.000 / 18 |
| BC | 6 | 4 | 0.939 / 0.615 | 1.000 / 1.000 / 0.595 / 0.000 / 74 | 1.000 / 1.000 / 0.595 / 0.000 / 74 | 1.000 / 1.000 / 0.563 / 0.000 / 70 |
| BC | 6 | 12 | 0.939 / 0.574 | 1.000 / 1.000 / 0.363 / 0.000 / 228 | 1.000 / 1.000 / 0.363 / 0.000 / 228 | 1.000 / 1.000 / 0.356 / 0.000 / 213 |
| D38 | 4 | 1 | 0.944 / 0.524 | 1.000 / 1.000 / 0.629 / 0.000 / 8 | 1.000 / 1.000 / 0.633 / 0.000 / 7 | 1.000 / 1.000 / 0.633 / 0.000 / 7 |
| D38 | 4 | 4 | 0.958 / 0.673 | 1.000 / 1.000 / 0.686 / 0.000 / 44 | 1.000 / 1.000 / 0.714 / 0.000 / 37 | 1.000 / 1.000 / 0.714 / 0.000 / 37 |
| D38 | 4 | 12 | 0.944 / 0.529 | 1.000 / 1.000 / 0.419 / 0.000 / 140 | 1.000 / 1.000 / 0.478 / 0.000 / 117 | 1.000 / 1.000 / 0.478 / 0.000 / 117 |
| D38 | 6 | 1 | 0.931 / 0.492 | 1.000 / 1.000 / 0.540 / 0.000 / 17 | 1.000 / 1.000 / 0.540 / 0.000 / 17 | 1.000 / 1.000 / 0.546 / 0.000 / 18 |
| D38 | 6 | 4 | 0.958 / 0.696 | 1.000 / 1.000 / 0.667 / 0.000 / 74 | 1.000 / 1.000 / 0.667 / 0.000 / 74 | 1.000 / 1.000 / 0.644 / 0.000 / 70 |
| D38 | 6 | 12 | 0.958 / 0.537 | 1.000 / 1.000 / 0.357 / 0.000 / 228 | 1.000 / 1.000 / 0.357 / 0.000 / 228 | 1.000 / 1.000 / 0.348 / 0.000 / 213 |

- H1 (noise-limited depth, margin; FakeAuckland, n=6): **CONFIRMED** (BC/RPSF: deepest 0.215, best 0.491 at L=4; BC/REC: deepest 0.215, best 0.512 at L=4; BC/L3T: deepest 0.215, best 0.515 at L=4; D38/RPSF: deepest 0.247, best 0.579 at L=4; D38/REC: deepest 0.246, best 0.593 at L=4; D38/L3T: deepest 0.252, best 0.598 at L=4)
- H2 (noise-limited depth, shot accuracy; FakeAuckland, n=6): **AMBIGUOUS** (BC/RPSF: deepest 0.992, best 1.000; BC/REC: deepest 0.987, best 1.000; BC/L3T: deepest 0.987, best 1.000; D38/RPSF: deepest 1.000, best 1.000; D38/REC: deepest 1.000, best 1.000; D38/L3T: deepest 1.000, best 1.000)
- H3 (REC keeps more margin than RPSF, pooled, every device): **CONFIRMED** (FakeAuckland REC-RPSF +0.0070; FakeTorino REC-RPSF +0.0150)
- H4 (REC level with L3T, pooled margin, every device): **CONFIRMED** (FakeAuckland REC-L3T +0.0070; FakeTorino REC-L3T +0.0055)
- H5 (REC flips no more predictions than RPSF, pooled, every device): **CONFIRMED** (FakeAuckland REC 0.0000 vs RPSF 0.0000; FakeTorino REC 0.0000 vs RPSF 0.0000)
- H6 (fine-tuning through the noise helps at L=12): **AMBIGUOUS** (FTN-DEP -0.0020, FT0-DEP -0.0102)
  - L=4: DEP {'acc': 1.0, 'margin': 0.5116422917827876, 'shot_acc': 1.0, 'ideal_acc': 1.0, 'ideal_margin': 0.6938123831358659}, FT0 {'acc': 1.0, 'margin': 0.5159532931397938, 'shot_acc': 1.0, 'ideal_acc': 1.0, 'ideal_margin': 0.6964939300496046}, FTN {'acc': 1.0, 'margin': 0.5159655652455682, 'shot_acc': 1.0, 'ideal_acc': 1.0, 'ideal_margin': 0.6961022753540002}
  - L=4: DEP {'acc': 1.0, 'margin': 0.5116422917827876, 'shot_acc': 1.0, 'ideal_acc': 1.0, 'ideal_margin': 0.6938123831358659}, FT0 {'acc': 1.0, 'margin': 0.519691668030069, 'shot_acc': 1.0, 'ideal_acc': 1.0, 'ideal_margin': 0.7085503326567678}, FTN {'acc': 1.0, 'margin': 0.5233923022701505, 'shot_acc': 1.0, 'ideal_acc': 1.0, 'ideal_margin': 0.7109189611101332}
  - L=12: DEP {'acc': 1.0, 'margin': 0.21500898852146103, 'shot_acc': 0.9916666666666667, 'ideal_acc': 1.0, 'ideal_margin': 0.6502311172311978}, FT0 {'acc': 1.0, 'margin': 0.20829952459279902, 'shot_acc': 0.9625, 'ideal_acc': 1.0, 'ideal_margin': 0.637818558800934}, FTN {'acc': 1.0, 'margin': 0.21461850635942836, 'shot_acc': 0.9916666666666667, 'ideal_acc': 1.0, 'ideal_margin': 0.6521255927153214}
  - L=12: DEP {'acc': 1.0, 'margin': 0.21500898852146103, 'shot_acc': 0.9916666666666667, 'ideal_acc': 1.0, 'ideal_margin': 0.6502311172311978}, FT0 {'acc': 0.9166666666666666, 'margin': 0.2012820550775923, 'shot_acc': 0.9166666666666666, 'ideal_acc': 1.0, 'ideal_margin': 0.6116710414516361}, FTN {'acc': 1.0, 'margin': 0.21134799233821452, 'shot_acc': 0.9625, 'ideal_acc': 1.0, 'ideal_margin': 0.6402529653018881}

GATE (stage 2): GO -- peak True, compiler difference True (cells with |shot acc REC - RPSF| >= 2 test points: none)

Reported without prediction -- the noise's share of the accuracy (FakeAuckland, n = 6): ideal accuracy minus shot accuracy, per L (points of the test set):
  - BC/RPSF: L=1 +0.1, L=4 -0.7, L=12 -0.6
  - BC/REC: L=1 +0.1, L=4 -0.7, L=12 -0.6
  - BC/L3T: L=1 +0.1, L=4 -0.7, L=12 -0.6
  - D38/RPSF: L=1 -0.8, L=4 -0.5, L=12 -0.5
  - D38/REC: L=1 -0.8, L=4 -0.5, L=12 -0.5
  - D38/L3T: L=1 -0.8, L=4 -0.5, L=12 -0.5

SUMMARY {"H1": "CONFIRMED", "H2": "AMBIGUOUS", "H3": "CONFIRMED", "H4": "CONFIRMED", "H5": "CONFIRMED", "H6": "AMBIGUOUS"}
