# HW2 scores (2026-10-13T10:04:09+00:00 (2026-10-13 12:04:09 CEST))

device ibm_marrakesh; calibration 2026-10-13 11:37:12+02:00; job db702a2mb58s738chiqg; usage {'qpu_charge_time_seconds': 102, 'value': 1.6968, 'unit': 'resource_units', 'status': 'complete', 'details': [{'value': 1.6968, 'unit': 'resource_units', 'metric': {'id': 'caffe3ca-729c-5750-8777-df63bbf8bfd9', 'value': 102}}]}

F: normalised linear XEB against the ideal distribution, mean of the two repeats.

| part | test | arm | 2q | ESP | duration us | idle-est | F rep 1 | F rep 2 | F mean (SE) |
|---|---|---|---|---|---|---|---|---|---|
| R | qaoa_n12_p2 | QK3 | 146 | 0.5229 | 7.85 | 0.2174 | 0.2910 | 0.3057 | 0.2983 (0.0063) |
| R | qaoa_n12_p2 | PSF | 141 | 0.6047 | 7.71 | 0.3105 | 0.4268 | 0.4076 | 0.4172 (0.0077) |
| R | qaoa_n20_p2 | QK3 | 251 | 0.3284 | 7.46 | 0.07176 | 0.2814 | 0.2718 | 0.2766 (0.0077) |
| R | qaoa_n20_p2 | PSF | 218 | 0.3317 | 8.04 | 0.02651 | 0.1675 | 0.1561 | 0.1618 (0.0056) |
| R | ba_n14 | QK3 | 391 | 0.2202 | 18.38 | 0.02192 | 0.2059 | 0.2010 | 0.2035 (0.0103) |
| R | ba_n14 | PSF | 287 | 0.3315 | 16.51 | 0.005358 | 0.2135 | 0.2393 | 0.2264 (0.0111) |
| R | ba_n16 | QK3 | 477 | 0.1699 | 21.12 | 0.008224 | 0.2032 | 0.2327 | 0.2179 (0.0089) |
| R | ba_n16 | PSF | 406 | 0.2861 | 20.78 | 0.004792 | 0.2782 | 0.2894 | 0.2838 (0.0098) |
| R | ba_n18 | QK3 | 584 | 0.09582 | 24.32 | 0.0002287 | 0.0731 | 0.0792 | 0.0762 (0.0061) |
| R | ba_n18 | PSF | 526 | 0.1385 | 28.62 | 0.001002 | 0.0540 | 0.0359 | 0.0449 (0.0058) |
| R | ba_n20 | QK3 | 668 | 0.06649 | 32.86 | 3.507e-06 | 0.0992 | 0.0882 | 0.0937 (0.0061) |
| R | ba_n20 | PSF | 638 | 0.07678 | 38.18 | 1.009e-07 | 0.0240 | 0.0277 | 0.0258 (0.0035) |
| R | tfim_n12 | QK3 | 82 | 0.7157 | 10.33 | 0.295 | 1.0947 | 1.1345 | 1.1146 (0.0194) |
| R | tfim_n12 | PSF | 64 | 0.7556 | 5.84 | 0.5527 | 1.2492 | 1.3281 | 1.2887 (0.0211) |
| R | tfim_n16 | QK3 | 114 | 0.6219 | 14.33 | 0.1408 | 0.6145 | 0.6210 | 0.6177 (0.0174) |
| R | tfim_n16 | PSF | 88 | 0.6719 | 7.91 | 0.342 | 0.5757 | 0.6005 | 0.5881 (0.0168) |
| R | ghz_n12 | QK3 | 11 | 0.8652 | 1.20 | 0.8012 | 0.8176 | 0.8177 | 0.8177 (0.0035) |
| R | ghz_n12 | PSF | 11 | 0.8652 | 1.20 | 0.8012 | 0.8162 | 0.8111 | 0.8137 (0.0036) |
| R | tfimP_n12 | QK3 | 82 | 0.7137 | 10.40 | 0.2922 | 0.5370 | 0.5401 | 0.5385 (0.0093) |
| R | tfimP_n12 | PSF | 64 | 0.751 | 5.88 | 0.516 | 0.7305 | 0.7265 | 0.7285 (0.0105) |
| R | tfimP_n16 | QK3 | 114 | 0.6192 | 14.40 | 0.1391 | 0.2407 | 0.2391 | 0.2399 (0.0062) |
| R | tfimP_n16 | PSF | 88 | 0.6675 | 7.95 | 0.2928 | 0.5543 | 0.5703 | 0.5623 (0.0097) |
| S | tfimP_n12 | S0 | 62 | 0.7514 | 7.92 | 0.3805 | 0.6364 | 0.6500 | 0.6432 (0.0100) |
| S | tfimP_n12 | S1 | 66 | 0.7457 | 1.64 | 0.6713 | 0.8350 | 0.8329 | 0.8340 (0.0113) |
| S | tfimP_n16 | S0 | 86 | 0.6675 | 10.92 | 0.215 | 0.5079 | 0.4975 | 0.5027 (0.0092) |
| S | tfimP_n16 | S1 | 90 | 0.661 | 1.72 | 0.5532 | 0.5937 | 0.5774 | 0.5856 (0.0095) |
| S | qaoa_n12_p2 | S0 | 141 | 0.6047 | 7.71 | 0.3105 | 0.4004 | 0.4075 | 0.4039 (0.0076) |
| S | qaoa_n12_p2 | S1 | 133 | 0.6169 | 6.91 | 0.3392 | 0.4066 | 0.4043 | 0.4055 (0.0075) |
| S | qaoa_n20_p2 | S0 | 218 | 0.3317 | 8.04 | 0.02651 | 0.1613 | 0.1537 | 0.1575 (0.0054) |
| S | qaoa_n20_p2 | S1 | 208 | 0.3399 | 7.91 | 0.03257 | 0.1437 | 0.1437 | 0.1437 (0.0052) |

## Part R: PSF / QK3

| test | ESP ratio | idle-est ratio | duration ratio | F ratio | F PSF | F QK3 | largest repeat difference |
|---|---|---|---|---|---|---|---|
| qaoa_n12_p2 | 1.156 | 1.429 | 0.982 | 1.398 | 0.4172 | 0.2983 | 0.0192 |
| qaoa_n20_p2 | 1.010 | 0.369 | 1.079 | 0.585 | 0.1618 | 0.2766 | 0.0114 |
| ba_n14 | 1.505 | 0.244 | 0.898 | 1.113 | 0.2264 | 0.2035 | 0.0258 |
| ba_n16 | 1.684 | 0.583 | 0.984 | 1.302 | 0.2838 | 0.2179 | 0.0296 |
| ba_n18 | 1.446 | 4.383 | 1.177 | 0.590 | 0.0449 | 0.0762 | 0.0182 |
| ba_n20 | 1.155 | 0.029 | 1.162 | 0.276 | 0.0258 | 0.0937 | 0.0110 |
| tfim_n12 | 1.056 | 1.873 | 0.565 | 1.156 | 1.2887 | 1.1146 | 0.0790 |
| tfim_n16 | 1.080 | 2.429 | 0.552 | 0.952 | 0.5881 | 0.6177 | 0.0249 |
| ghz_n12 | 1.000 | 1.000 | 1.000 | 0.995 | 0.8137 | 0.8177 | 0.0052 |
| tfimP_n12 | 1.052 | 1.766 | 0.565 | 1.353 | 0.7285 | 0.5385 | 0.0039 |
| tfimP_n16 | 1.078 | 2.105 | 0.552 | 2.344 | 0.5623 | 0.2399 | 0.0160 |

### Predictions, part R

- R1 (ESP >= 1.2 x QK3 does not give the higher F on more than half): 3 tests, PSF's F >= QK3's on 2 -> REFUTED
- R2 (ESP does not rank the pair: rank correlation <= 0.3): +0.429 over 7 -> REFUTED
- R3 (the idle-aware estimate: same winner on >= 5 of 7 and rank correlation above ESP's): same winner 4 (ESP 4); rank correlation +0.464 (ESP +0.429) -> REFUTED
- R4 (where PSF's duration <= 0.8 x QK3's, PSF's F >= QK3's): 3 tests (tfim_n12, tfimP_n12, tfimP_n16); -> CONFIRMED
- R5 (Barabasi-Albert: PSF's F <= 1.05 x QK3's on >= 3 of 4): 2 of 4 -> REFUTED
- R6 (GHZ control: |F ratio - 1| <= 0.10): CONFIRMED
- **Replication (R1, R3 and R4 all confirmed): NOT REPLICATED**
- tfim_n16 (reported, not read: XEB failed there in HW1): F PSF 0.5881, QK3 0.6177

## Part S: S1 (item 83 on) / S0 (off), the same gate blocks

| test | 2q S1 / S0 | ESP ratio | idle-est ratio | duration ratio | F ratio | F S1 | F S0 | largest repeat difference |
|---|---|---|---|---|---|---|---|---|
| tfimP_n12 | 66 / 62 | 0.992 | 1.764 | 0.207 | 1.297 | 0.8340 | 0.6432 | 0.0136 |
| tfimP_n16 | 90 / 86 | 0.990 | 2.573 | 0.157 | 1.165 | 0.5856 | 0.5027 | 0.0163 |
| qaoa_n12_p2 | 133 / 141 | 1.020 | 1.092 | 0.897 | 1.004 | 0.4055 | 0.4039 | 0.0071 |
| qaoa_n20_p2 | 208 / 218 | 1.025 | 1.228 | 0.983 | 0.912 | 0.1437 | 0.1575 | 0.0076 |

### Predictions, part S

- S1 (TFIM from a product state: S1's F > S0's on every pair): 2 of 2 -> CONFIRMED
- S2 (TFIM: geometric mean F ratio S1 / S0 >= 1.05): 1.229 -> CONFIRMED
- S3 (reported): QAOA pairs: qaoa_n12_p2 duration 0.897, F ratio 1.004; qaoa_n20_p2 duration 0.983, F ratio 0.912
- S4 (reported): which estimate picked the device's winner on each pair: tfimP_n12: ESP no, idle-est yes; tfimP_n16: ESP no, idle-est yes; qaoa_n12_p2: ESP yes, idle-est yes; qaoa_n20_p2: ESP no, idle-est no
- reported: F / ESP per arm (geometric mean): QK3 0.879, PSF 0.710, S0 0.672, S1 0.724
