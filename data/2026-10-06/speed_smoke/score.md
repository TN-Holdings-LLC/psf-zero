# SPEED score (SMOKE)

P0: PASS -- files 6 of 6; errors 0; inexact a12 outputs 0; off-target a12 outputs 0; counts ok True

| device | identical | median a11 s | median a12 s | ratio of medians | median of ratios |
|---|---|---|---|---|---|
| FakeTorino | 32/32 | 0.712 | 0.456 | 0.640 | 0.610 |
| FakeKingston | 32/32 | 0.990 | 0.569 | 0.575 | 0.574 |
| FakeAuckland | 32/32 | 0.188 | 0.170 | 0.905 | 0.894 |
| FakeHanoiV2 | 32/32 | 0.182 | 0.166 | 0.911 | 0.904 |
| FakeBrussels | 32/32 | 0.442 | 0.319 | 0.722 | 0.763 |
| FakeOsaka | 32/32 | 0.456 | 0.357 | 0.782 | 0.786 |

## Predictions

- S1 (a12 returns a11's circuit (identical on every circuit of every device; refuted if any differs)): **CONFIRMED** -- {"FakeTorino": 1.0, "FakeKingston": 1.0, "FakeAuckland": 1.0, "FakeHanoiV2": 1.0, "FakeBrussels": 1.0, "FakeOsaka": 1.0}
- S2 (faster on the cz devices (median a12 / median a11 <= 0.75 on both; refuted > 1.00 on either)): **CONFIRMED** -- {"FakeTorino": 0.6401266062643431, "FakeKingston": 0.5748851068127871}
- S3 (not slower elsewhere (median ratio <= 0.90 on the cx and ecr devices; refuted > 1.05 on any)): **AMBIGUOUS** -- {"FakeAuckland": 0.9047403823146253, "FakeHanoiV2": 0.9111068472123237, "FakeBrussels": 0.7220323252444767, "FakeOsaka": 0.7823983146622192}

Reported: median of per-circuit ratios a12 / a11: {"FakeTorino": 0.61, "FakeKingston": 0.574, "FakeAuckland": 0.894, "FakeHanoiV2": 0.904, "FakeBrussels": 0.763, "FakeOsaka": 0.786}
