# MARGIN score

P0: PASS -- files 96/96, training files 4/4; max state infidelity 2.09e-09 (<= 1e-06); max |reduced - whole-device| 0.00e+00 over 1152 circuits; versions per arm True; files as locked True; SWITCH_MARGIN 0.05 in C24 only True; every stale Target differs from the true one True; seeds [(5, 5)]

- M1 (with stale calibrations C24 keeps at least the guarded call's margin: C24 - RPSF >= 0 on both devices (REFUTED if < -0.002 on either)): **CONFIRMED** (FakeAuckland C24-RPSF stale +0.0009 (per draw [0.0048, -0.0017, -0.0005]; REC-RPSF +0.0010); FakeTorino C24-RPSF stale +0.0221 (per draw [0.0219, 0.0224, 0.022]; REC-RPSF +0.0245))
- M2 (with stale calibrations the margin costs nothing: C24 - REC >= -0.001 on both devices (REFUTED if < -0.005 on either)): **AMBIGUOUS** (FakeAuckland C24-REC stale -0.0002 (per draw [-0.0031, 0.002, 0.0005]); FakeTorino C24-REC stale -0.0024 (per draw [-0.0006, -0.0058, -0.0009]))
- M3 (with the true calibration it costs little: REC - C24 <= +0.002 on both devices (REFUTED if > +0.005 on either)): **AMBIGUOUS** (FakeAuckland REC-C24 true +0.0034 (REC-RPSF true +0.0058); FakeTorino REC-C24 true +0.0000 (REC-RPSF true +0.0175))
- M4 (FakeTorino's structural lead is kept: C24 - RPSF >= +0.01 with the true and with stale calibrations (REFUTED if < +0.005 in either)): **CONFIRMED** (FakeTorino C24-RPSF true +0.0175, stale +0.0221)
- M5 (no bad draw: FakeAuckland's worst stale draw C24 - RPSF >= -0.005 (REFUTED if < -0.01)): **CONFIRMED** (FakeAuckland worst draw C24-RPSF -0.0017)

Reading rule (Addendum 405): where C24 returns REC's circuit on more than 95% of the stale rows of a device, M1, M2 and M5 on that device say that the margin rarely bit, not that it works. Share of stale rows where C24's circuit differs from REC's: FakeAuckland 0.306, FakeTorino 0.174

ITEM 51 (decision rule of Addendum 405: P0, M1 CONFIRMED, M2 and M3 not REFUTED, the margin bit on at least one device): PROPOSE ACCEPTANCE

Reported without prediction -- pooled margin, accuracy at fixed shot budgets (exact binomial, as Addendum 401), flips, two-qubit gates on failed couplers of the true device, compile time, and how often the margin turned an alternative away (C24's counters, summed over its files):

## FakeAuckland, true calibration

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | flips | gates on failed couplers | compile s (median) |
|---|---|---|---|---|---|---|---|---|
| REC | 0.4275 | 0.8845 | 0.9395 | 0.9573 | 0.9623 | 12.0 | 0 | 0.860 |
| C24 | 0.4241 | 0.8828 | 0.9382 | 0.9568 | 0.9622 | 14.0 | 0 | 0.942 |
| RPSF | 0.4217 | 0.8825 | 0.9381 | 0.9568 | 0.9622 | 14.0 | 0 | 0.115 |

## FakeAuckland, stale (mean of 3 draws) calibration

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | flips | gates on failed couplers | compile s (median) |
|---|---|---|---|---|---|---|---|---|
| REC | 0.4091 | 0.8742 | 0.9339 | 0.9553 | 0.9612 | 15.7 | 0 | 1.138 |
| C24 | 0.4089 | 0.8744 | 0.9336 | 0.9551 | 0.9611 | 16.0 | 0 | 0.911 |
| RPSF | 0.4080 | 0.8731 | 0.9327 | 0.9548 | 0.9611 | 15.0 | 0 | 0.119 |

FakeAuckland: C24's circuit differs from REC's on 0.401 of the true-calibration rows and 0.306 of the stale rows; the margin turned away 3191 candidates of item 37 and 1401 re-syntheses (C24's choices: psf 6417, floor 1080, level3 1431)

## FakeTorino, true calibration

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | flips | gates on failed couplers | compile s (median) |
|---|---|---|---|---|---|---|---|---|
| REC | 0.5132 | 0.9194 | 0.9540 | 0.9623 | 0.9645 | 5.0 | 0 | 0.767 |
| C24 | 0.5132 | 0.9194 | 0.9540 | 0.9623 | 0.9645 | 5.0 | 0 | 0.743 |
| RPSF | 0.4957 | 0.9128 | 0.9528 | 0.9621 | 0.9646 | 5.0 | 0 | 0.134 |

## FakeTorino, stale (mean of 3 draws) calibration

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | flips | gates on failed couplers | compile s (median) |
|---|---|---|---|---|---|---|---|---|
| REC | 0.5056 | 0.9188 | 0.9532 | 0.9620 | 0.9643 | 6.7 | 0 | 0.862 |
| C24 | 0.5031 | 0.9172 | 0.9528 | 0.9618 | 0.9642 | 7.0 | 0 | 0.840 |
| RPSF | 0.4810 | 0.9099 | 0.9513 | 0.9615 | 0.9643 | 7.7 | 0 | 0.125 |

FakeTorino: C24's circuit differs from REC's on 0.013 of the true-calibration rows and 0.174 of the stale rows; the margin turned away 1449 candidates of item 37 and 1160 re-syntheses (C24's choices: psf 4831, floor 164, level3 3933)

SUMMARY {"M1": "CONFIRMED", "M2": "AMBIGUOUS", "M3": "AMBIGUOUS", "M4": "CONFIRMED", "M5": "CONFIRMED", "P0": "PASS", "ITEM51": "PROPOSE"}
