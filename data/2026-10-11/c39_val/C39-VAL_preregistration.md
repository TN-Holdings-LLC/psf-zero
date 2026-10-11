# C39-VAL -- fixed before the run (local; to be recorded in an Addendum with the results)

Written 2026-10-11 09:44 CEST, before the full run; only a smoke run of 4 jobs (one tsp-type
circuit, k = 3, on four devices) had been seen.

- Candidate: psf_compile 2026-10-11.c39 = release 2026-10-11.2 + items 69-70 (sha256 7f0207a4dcd1fba6d4f98fda72d93b434ccf1ab5d962bab27b5096c7d9e65615), built by make_c39.py;
  test_c39.py (sha256 c6a5920b6dbc5447cf38ffb5b9e4015206e0fb8aa41c4b8e9dc19bd7c51f07aa): 8 tests passed.
- Script: c39_val.py (sha256 b39f7c0524530ecb1e1580f7951026fbdcf813b998aeb8211bada490199485b2). Circuits all generated with new seeds ("C39-VAL|..."): tsp-type in unary encoding
  (9, 16, 25 qubits; 9 circuits), dense random graphs G(n, 0.4) (6), MaxCut type on 3-regular graphs (6), 3-SAT type
  (4), and non-commuting controls (3). Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen,
  FakeBoston. Arms: QK3 (level 3, seed 0), R1 (release 2026-10-11.1), R2 (release 2026-10-11.2), C39.
- Runnable: best ESP of the four >= 0.01; ESP of an output on a failed element = 0.

| ID | prediction, per device | confirmed | refuted |
|---|---|---|---|
| X1 | C39 outputs on a failed element | 0 | any |
| X2 | non-commuting inputs: C39's output = R1's = R2's, by value | all | any differs |
| X3 | every commuting test: ESP C39 >= ESP R1 (by design of item 69) | all | any lower |
| X4 | every commuting test: ESP C39 >= ESP R2 (by design of item 69) | all | any lower |
| X5 | runnable commuting tests: ESP C39 / R1 (geometric mean) | >= 1.10 | < 1.00 |
| X6 | runnable tsp-type tests: ESP C39 / R1 (reported with R2 / R1) | >= 1.00 | < 1.00 |
| X7 | median compile time of C39 on commuting tests (reported) | <= 1.0 s | -- |

Proposed release rule (the owner decides): X1-X4 confirmed on every device and X5 not refuted on any.
