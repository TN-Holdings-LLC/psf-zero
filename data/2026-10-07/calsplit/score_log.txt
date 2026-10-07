# CALSPLIT score

P0: PASS -- files 88/88; max state infidelity 3.32e-09 (<= 1e-06); max |reduced - whole-device| 0.00e+00 over 1056 circuits; versions ['2026-10-07.1']; release as locked True; every stale Target differs from the true one True

- K1 (calibration still helps: REC - better blind arm pooled margin >= +0.002 on both devices): **CONFIRMED** (FakeAuckland REC-DEF +0.0308 (DEF 0.3693, L3B 0.3680); FakeTorino REC-L3B +0.0860 (DEF 0.2839, L3B 0.3978))
- K2 (the advantage over the guarded call survives: REC - RPSF >= +0.003 on both devices): **REFUTED** (FakeAuckland REC-RPSF -0.0041 (per draw [0.001, -0.0142, 0.0009]); FakeTorino REC-RPSF +0.0194 (per draw [-0.0005, 0.0314, 0.0273]))
- K3 (it shrinks by at most 0.005 against DEPTH-R's same-calibration value, on both devices): **AMBIGUOUS** (FakeAuckland same-calibration +0.0056 minus stale -0.0041 = +0.0097; FakeTorino same-calibration +0.0178 minus stale +0.0194 = -0.0016)
- K4 (REC level with L3T (both stale): |REC - L3T| <= 0.01 on both devices): **CONFIRMED** (FakeAuckland REC-L3T +0.0049; FakeTorino REC-L3T -0.0098)
- K5 (no more flipped answers than the better blind arm: flip-rate difference <= +0.002): **CONFIRMED** (FakeAuckland flips REC 18.7, DEF 24 (DEF 24, L3B 24) of 2232 (rate difference -0.0024); FakeTorino flips REC 8.3, L3B 19 (DEF 369, L3B 19) of 2232 (rate difference -0.0048))

Reported without prediction -- pooled margin per arm, accuracy at fixed shot budgets (exact binomial, as Addendum 401), two-qubit gates on failed couplers of the true device:

## FakeAuckland

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | gates on failed couplers |
|---|---|---|---|---|---|---|
| REC | 0.4001 | 0.8666 | 0.9178 | 0.9315 | 0.9341 | 0 |
| RPSF | 0.4042 | 0.8700 | 0.9190 | 0.9314 | 0.9337 | 0 |
| L3T | 0.3952 | 0.8648 | 0.9171 | 0.9311 | 0.9338 | 0 |
| DEF | 0.3693 | 0.8492 | 0.9091 | 0.9286 | 0.9332 | 0 |
| L3B | 0.3680 | 0.8464 | 0.9068 | 0.9279 | 0.9327 | 0 |

## FakeTorino

| arm | margin | acc at 15 shots | acc at 63 shots | acc at 255 shots | acc at 1023 shots | gates on failed couplers |
|---|---|---|---|---|---|---|
| REC | 0.4838 | 0.9031 | 0.9309 | 0.9351 | 0.9356 | 0 |
| RPSF | 0.4645 | 0.8972 | 0.9299 | 0.9350 | 0.9357 | 0 |
| L3T | 0.4936 | 0.9049 | 0.9311 | 0.9350 | 0.9355 | 0 |
| DEF | 0.2839 | 0.7386 | 0.7611 | 0.7651 | 0.7656 | 21204 |
| L3B | 0.3978 | 0.8541 | 0.9036 | 0.9243 | 0.9318 | 0 |

SUMMARY {"K1": "CONFIRMED", "K2": "REFUTED", "K3": "AMBIGUOUS", "K4": "CONFIRMED", "K5": "CONFIRMED"}
