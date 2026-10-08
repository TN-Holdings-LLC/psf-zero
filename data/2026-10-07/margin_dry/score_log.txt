# MARGIN score (DRY RUN -- not a result)

P0: PASS -- files 96/96, training files 4/4; max state infidelity 5.11e-15 (<= 1e-06); max |reduced - whole-device| 0.00e+00 over 576 circuits; versions per arm True; files as locked True; SWITCH_MARGIN 0.05 in C24 only True; every stale Target differs from the true one True; seeds [(2, 2)]

- M1 (with stale calibrations C24 keeps at least the guarded call's margin: C24 - RPSF >= 0 on both devices (REFUTED if < -0.002 on either)): **CONFIRMED** (FakeAuckland C24-RPSF stale +0.0013 (per draw [0.0028, -0.0001, 0.0012]; REC-RPSF +0.0019); FakeTorino C24-RPSF stale +0.0188 (per draw [0.0199, 0.0186, 0.0178]; REC-RPSF +0.0193))
- M2 (with stale calibrations the margin costs nothing: C24 - REC >= -0.001 on both devices (REFUTED if < -0.005 on either)): **CONFIRMED** (FakeAuckland C24-REC stale -0.0006 (per draw [-0.0039, 0.0023, -0.0003]); FakeTorino C24-REC stale -0.0005 (per draw [-0.0002, -0.0014, 0.0]))
- M3 (with the true calibration it costs little: REC - C24 <= +0.002 on both devices (REFUTED if > +0.005 on either)): **REFUTED** (FakeAuckland REC-C24 true +0.0073 (REC-RPSF true +0.0075); FakeTorino REC-C24 true +0.0001 (REC-RPSF true +0.0148))
- M4 (FakeTorino's structural lead is kept: C24 - RPSF >= +0.01 with the true and with stale calibrations (REFUTED if < +0.005 in either)): **CONFIRMED** (FakeTorino C24-RPSF true +0.0147, stale +0.0188)
- M5 (no bad draw: FakeAuckland's worst stale draw C24 - RPSF >= -0.005 (REFUTED if < -0.01)): **CONFIRMED** (FakeAuckland worst draw C24-RPSF -0.0001)

Reading rule (Addendum 405): where C24 returns REC's circuit on more than 95% of the stale rows of a device, M1, M2 and M5 on that device say that the margin rarely bit, not that it works. Share of stale rows where C24's circuit differs from REC's: FakeAuckland 0.319, FakeTorino 0.053

ITEM 51 (decision rule of Addendum 405: P0, M1 CONFIRMED, M2 and M3 not REFUTED, the margin bit on at least one device): DO NOT PROPOSE

Reported without prediction -- pooled margin, accuracy at fixed shot budgets (exact binomial, as Addendum 401), flips, two-qubit gates on failed couplers of the true device, compile time, and how often the margin turned an alternative away (C24's counters, summed over its files):

## FakeAuckland, true calibration

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | flips | gates on failed couplers | compile s (median) |
|---|---|---|---|---|---|---|---|---|
| REC | 0.4548 | 0.9122 | 0.9649 | 0.9815 | 0.9856 | 0.0 | 0 | 0.415 |
| C24 | 0.4475 | 0.9092 | 0.9641 | 0.9816 | 0.9857 | 0.0 | 0 | 0.387 |
| RPSF | 0.4473 | 0.9093 | 0.9641 | 0.9817 | 0.9858 | 0.0 | 0 | 0.054 |

## FakeAuckland, stale (mean of 3 draws) calibration

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | flips | gates on failed couplers | compile s (median) |
|---|---|---|---|---|---|---|---|---|
| REC | 0.4360 | 0.9020 | 0.9597 | 0.9801 | 0.9849 | 0.0 | 0 | 0.459 |
| C24 | 0.4353 | 0.9024 | 0.9601 | 0.9804 | 0.9853 | 0.0 | 0 | 0.396 |
| RPSF | 0.4341 | 0.9015 | 0.9597 | 0.9800 | 0.9850 | 0.0 | 0 | 0.049 |

FakeAuckland: C24's circuit differs from REC's on 0.319 of the true-calibration rows and 0.319 of the stale rows; the margin turned away 197 candidates of item 37 and 28 re-syntheses (C24's choices: psf 470, floor 54, level3 52)

## FakeTorino, true calibration

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | flips | gates on failed couplers | compile s (median) |
|---|---|---|---|---|---|---|---|---|
| REC | 0.5225 | 0.9278 | 0.9692 | 0.9837 | 0.9873 | 0.0 | 0 | 0.569 |
| C24 | 0.5224 | 0.9277 | 0.9691 | 0.9837 | 0.9873 | 0.0 | 0 | 0.596 |
| RPSF | 0.5077 | 0.9242 | 0.9685 | 0.9837 | 0.9874 | 0.0 | 0 | 0.094 |

## FakeTorino, stale (mean of 3 draws) calibration

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | flips | gates on failed couplers | compile s (median) |
|---|---|---|---|---|---|---|---|---|
| REC | 0.5158 | 0.9345 | 0.9724 | 0.9847 | 0.9874 | 0.0 | 0 | 0.712 |
| C24 | 0.5153 | 0.9341 | 0.9723 | 0.9848 | 0.9876 | 0.0 | 0 | 0.693 |
| RPSF | 0.4965 | 0.9284 | 0.9712 | 0.9844 | 0.9875 | 0.0 | 0 | 0.133 |

FakeTorino: C24's circuit differs from REC's on 0.028 of the true-calibration rows and 0.053 of the stale rows; the margin turned away 33 candidates of item 37 and 42 re-syntheses (C24's choices: psf 330, floor 20, level3 226)

SUMMARY {"M1": "CONFIRMED", "M2": "CONFIRMED", "M3": "REFUTED", "M4": "CONFIRMED", "M5": "CONFIRMED", "P0": "PASS", "ITEM51": "NO"}
