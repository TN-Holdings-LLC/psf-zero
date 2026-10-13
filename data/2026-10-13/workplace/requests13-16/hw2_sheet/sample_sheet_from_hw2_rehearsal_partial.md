# HW2 results sheet (assembled 2026-10-13 09:48:21 CEST by hw2_sheet.py; the numbers are the tools' own)

**Run**

- **Device:** fake_fez (device rule: [{'name': 'FakeFez', 'fake': True}]); calibration 2025-02-26 15:16:25-05:00.
- **Prepare:** manifest written 2026-10-13T07:42:08+00:00 (2026-10-13 09:42:08 CEST); 30 compiled circuits, 30 submitted (60 PUBs with the second repeat); not submitted: none.
- **Checks:** ISA and failed elements passed on every submitted circuit (30 of 30); exact (noiseless, <= 24 active qubits) largest difference 5.8e-16; above 24 active qubits (n/a): 0.
- **Job:** rehearsal; submitted None; fetched 2026-10-13T07:42:21+00:00 (2026-10-13 09:42:21 CEST); status DONE (local); usage {'quantum_seconds': 0}.

**Part R (F: normalised linear XEB, mean of two repeats):**

| test | ESP QK3 | ESP PSF | ESP ratio PSF/QK3 | idle-est ratio | duration PSF / QK3 (us) | F QK3 | F PSF | **F ratio** | largest repeat difference |
|---|---|---|---|---|---|---|---|---|---|
| qaoa_n12_p2 | 0.557 | 0.578 | 1.037 | 1.090 | 6.4 / 8.0 | 1.001 | 1.013 | **1.013** | 0.020 |
| qaoa_n20_p2 | 0.269 | 0.284 | 1.055 | 0.492 | 11.2 / 7.6 | 0.993 | 0.989 | **0.995** | 0.032 |
| ba_n14 | 0.235 | 0.296 | 1.261 | 1.302 | 19.7 / 18.8 | 0.982 | 0.987 | **1.005** | 0.039 |
| ba_n16 | 0.170 | 0.204 | 1.203 | 0.757 | 28.3 / 21.6 | 0.979 | 0.987 | **1.009** | 0.057 |
| ba_n18 | 0.113 | 0.138 | 1.218 | 0.119 | 27.3 / 24.7 | 1.000 | 0.995 | **0.995** | 0.026 |
| ba_n20 | 0.055 | 0.088 | 1.597 | 0.154 | 30.2 / 33.6 | 0.982 | 0.998 | **1.017** | 0.104 |
| tfim_n12 | 0.688 | 0.737 | 1.071 | 1.811 | 5.6 / 9.9 | 1.020 | 1.007 | **0.987** | 0.074 |
| tfim_n16 | 0.584 | 0.650 | 1.113 | 3.290 | 7.5 / 13.7 | 0.979 | 0.976 | **0.997** | 0.039 |
| ghz_n12 | 0.884 | 0.884 | 1.000 | 1.000 | 1.2 / 1.2 | 1.000 | 1.000 | **1.000** | 0.000 |
| tfimP_n12 | 0.686 | 0.734 | 1.070 | 1.659 | 5.6 / 9.9 | 1.011 | 0.984 | **0.973** | 0.045 |
| tfimP_n16 | 0.583 | 0.646 | 1.110 | 2.713 | 7.5 / 13.7 | 1.010 | 0.998 | **0.988** | 0.039 |

| ID | prediction | result |
|---|---|---|
| R1 | of the reading tests whose ESP ratio PSF / QK3 is >= 1.2, PSF's F >= QK3's on at most half | **REFUTED**: 3 of 4 (ba_n14 1.01, ba_n16 1.01, ba_n18 0.99, ba_n20 1.02) |
| R2 | rank correlation of ESP ratio and F ratio over the reading tests <= 0.3 | **CONFIRMED**: +0.214 (7 tests) |
| R3 | the estimate with idling picks the same winner as F on >= 5 of 7 reading tests, and its rank correlation with the F ratio is above ESP's | **REFUTED**: same winner 4 of 7 (ESP 4); rho -0.214 vs ESP +0.214 |
| R4 | on every test (the seven and the two tfimP) whose PSF duration is <= 0.8 x QK3's, PSF's F >= QK3's (at least two such tests) | **REFUTED**: tfim_n12 duration 0.57 F 0.99, tfimP_n12 duration 0.57 F 0.97, tfimP_n16 duration 0.55 F 0.99 |
| R5 | Barabasi-Albert: PSF's F <= 1.05 x QK3's on >= 3 of 4 | **CONFIRMED**: ba_n14 1.01, ba_n16 1.01, ba_n18 0.99, ba_n20 1.02 |
| R6 | GHZ control: \|F ratio - 1\| <= 0.10 | **CONFIRMED** (circuits differ): F ratio 1.000 |

**Replication rule (R1, R3 and R4 all confirmed): NOT REPLICATED.**

tfim_n16 is reported and not read (XEB failed there in HW1): F PSF 0.976, QK3 0.979.

**Part S (S1: item 83 on; S0: off; the same gate blocks):**

| test | 2q S1 / S0 | ESP ratio | idle-est ratio | duration ratio | F S0 | F S1 | **F ratio S1/S0** | largest repeat difference |
|---|---|---|---|---|---|---|---|---|
| tfimP_n12 | 66 / 62 | 0.988 | 1.834 | 0.201 | 0.984 | 1.015 | **1.031** | 0.023 |
| tfimP_n16 | 90 / 88 | 0.989 | 2.360 | 0.201 | 1.002 | 1.016 | **1.014** | 0.015 |
| qaoa_n12_p2 | 132 / 136 | 1.011 | 1.257 | 0.765 | 1.011 | 0.982 | **0.972** | 0.021 |
| qaoa_n20_p2 | 244 / 252 | 1.030 | 1.258 | 0.905 | 0.985 | 1.006 | **1.022** | 0.023 |

| ID | prediction | result |
|---|---|---|
| S1 | tfimP: S1's F > S0's on every pair | **CONFIRMED**: 2 of 2 |
| S2 | tfimP: geometric mean F ratio S1 / S0 >= 1.05 | **REFUTED**: 1.023 |
| S3 | reported: the QAOA pairs | qaoa_n12_p2 duration 0.765, F ratio 0.972; qaoa_n20_p2 duration 0.905, F ratio 1.022 |
| S4 | reported: which estimate picked the device's winner | tfimP_n12: ESP no, idle-est yes; tfimP_n16: ESP no, idle-est yes; qaoa_n12_p2: ESP no, idle-est no; qaoa_n20_p2: ESP yes, idle-est yes |

**Predictors on the seven reading pairs (predicted ratio PSF / QK3; device: F ratio):**

| test | device | ESP | R3 | NS2 |
|---|---|---|---|---|
| qaoa_n12_p2 | **1.013** | 1.037 | 1.090 | 1.013 |
| qaoa_n20_p2 | **0.995** | 1.055 | 0.492 | n/a |
| ba_n14 | **1.005** | 1.261 | 1.302 | 0.960 |
| ba_n16 | **1.009** | 1.203 | 0.757 | 0.975 |
| ba_n18 | **0.995** | 1.218 | 0.119 | n/a |
| ba_n20 | **1.017** | 1.597 | 0.154 | n/a |
| tfim_n12 | **0.987** | 1.071 | 1.811 | 1.027 |
| same winner | | 4 of 7 | 4 of 7 | 1 of 4 |
| rank correlation | | +0.21 | -0.21 | -0.20 |

Not available: P1, P2, P3, P4, P1c, P2c.

| ID | prediction | result |
|---|---|---|
| NS2a-e | | not available |
| Q1-Q3 | | not available |
| Q4 (amendment 2), Q2c | | not available |

(HW2-PLUS's own Q4 is "reported: each predictor's winner count and rank correlation" -- the table above; the amendment's Q4 is labelled "Q4 (amendment 2)" here to keep the two apart.)

**Fixed within-test analysis (plan v2):** not available.

**Inputs (SHA-256):**

| file | sha256 |
|---|---|
| manifest: hw2_manifest.json | `b2576ee9e7f8463644c93ed818a3806fcac1e88ee2ecfd864cc03d4c0d11a92c` |
| score: hw2_score.json | `ead42d7bff047fb5841617ff6beee39352d8835bf11016b84ffa3b7afdf3535c` |
| raw: hw2_raw_counts.json | `c89f6669e5654691ea00eab7e35ec02a08225d8cfdfff753d65db4a752d4b41f` |
| ns2_rows: ns_hw2.jsonl | `746a3e7cf7189eafa8c098f453969a0a20dcfe64f8d6a4917739b4fc84021e1c` |
| ns2_score | not available |
| pred_rows | not available |
| pred_score | not available |
| predc_rows | not available |
| predc_score | not available |
| within | not available |
| within_v1 | not available |
