# CALSPLIT score (DRY RUN -- not a result)

P0: PASS -- files 88/88; max state infidelity 6.83e-11 (<= 1e-06); max |reduced - whole-device| 0.00e+00 over 528 circuits; versions ['2026-10-07.1']; release as locked True; every stale Target differs from the true one True

- K1 (calibration still helps: REC - DEF pooled margin >= +0.002 on both devices): **CONFIRMED** (FakeAuckland REC-DEF +0.0305; FakeTorino REC-DEF +0.2064)
- K2 (the advantage over the guarded call survives: REC - RPSF >= +0.003 on both devices): **REFUTED** (FakeAuckland REC-RPSF -0.0029 (per draw [0.0003, -0.0101, 0.0011]); FakeTorino REC-RPSF +0.0140 (per draw [-0.0009, 0.0213, 0.0214]))
- K3 (it shrinks by at most 0.005 against DEPTH-R's same-calibration value, on both devices): **AMBIGUOUS** (FakeAuckland same-calibration +0.0070 minus stale -0.0029 = +0.0099; FakeTorino same-calibration +0.0150 minus stale +0.0140 = +0.0010)
- K4 (REC level with L3T (both stale): |REC - L3T| <= 0.01 on both devices): **CONFIRMED** (FakeAuckland REC-L3T +0.0059; FakeTorino REC-L3T -0.0076)
- K5 (no more flipped answers than the blind default call: flip-rate difference <= +0.002): **CONFIRMED** (FakeAuckland flips REC 0.0 DEF 0 of 144 (rate difference +0.0000); FakeTorino flips REC 0.0 DEF 32 of 144 (rate difference -0.2222))

Reported without prediction -- pooled margin per arm, accuracy at fixed shot budgets (exact binomial, as Addendum 401), two-qubit gates on failed couplers of the true device:

## FakeAuckland

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | gates on failed couplers |
|---|---|---|---|---|---|---|
| REC | 0.4413 | 0.9054 | 0.9621 | 0.9813 | 0.9858 | 0 |
| RPSF | 0.4442 | 0.9070 | 0.9629 | 0.9814 | 0.9857 | 0 |
| L3T | 0.4353 | 0.9028 | 0.9607 | 0.9805 | 0.9853 | 0 |
| DEF | 0.4108 | 0.8898 | 0.9539 | 0.9779 | 0.9841 | 0 |
| L3B | 0.4078 | 0.8836 | 0.9491 | 0.9764 | 0.9835 | 0 |

## FakeTorino

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | gates on failed couplers |
|---|---|---|---|---|---|---|
| REC | 0.5089 | 0.9331 | 0.9725 | 0.9851 | 0.9878 | 0 |
| RPSF | 0.4950 | 0.9276 | 0.9712 | 0.9846 | 0.9876 | 0 |
| L3T | 0.5165 | 0.9344 | 0.9726 | 0.9855 | 0.9881 | 0 |
| DEF | 0.3026 | 0.7578 | 0.7771 | 0.7828 | 0.7838 | 1080 |
| L3B | 0.4363 | 0.8956 | 0.9546 | 0.9796 | 0.9861 | 0 |

SUMMARY {"K1": "CONFIRMED", "K2": "REFUTED", "K3": "AMBIGUOUS", "K4": "CONFIRMED", "K5": "CONFIRMED"}
