# C40-VAL -- fixed before the run (local; to be recorded in an Addendum with the results)

Written 2026-10-11 10:14 CEST, before the full run; only a smoke run of 4 jobs (QAOA n = 12, p = 1 and one random
control, on FakeTorino and FakeKingston) had been seen.

- Prototype: psf_compile 2026-10-11.c40 = c39 + items 71-72 (sha256 9ab1478d064e54c66b32fb1fec27eb5d3218e991aab67f27bc3ea4dd8e321411),
  built by make_c40.py (sha256 39af448d91743530272ec1350c62f2484c4741df644891ae6d689d53deb7d6b9) from c39 (sha256 7f0207a4...5615);
  test_c40.py (sha256 92173c0d6fabb91d3daa3ece98ce70ff715e238c9fcb3d077d2b8c05831bbcec): 7 tests passed.
- Script: c40_val.py (sha256 baa3a7a104e377e717662f2bbec03eb65667a4147b448b70d2bbd887431498ee). All circuits generated with new
  seeds ("C40-VAL|..."), final measurements on all qubits; 27 tests x 6 devices = 162 jobs:
  - qaoa: QAOAAnsatz, MaxCut on random 3-regular graphs, n = 12, 20, 28, p = 1-3 (9);
  - trotter_noncommuting: transverse-field Ising on 3-regular graphs, LieTrotter 2 steps and SuzukiTrotter order 2,
    n = 10, 16, 22 (6);
  - trotter_commuting: longitudinal Ising LieTrotter 3 steps n = 12, 20; 3-SAT type LieTrotter 2 steps n = 14 (3);
  - single_noncommuting: Heisenberg on 3-regular graphs n = 10, 16; random 2-/3-local Paulis n = 10, 16; XY ring
    n = 12 (5);
  - rotation_gates: H layer, then 2 x (rzz on a 3-regular graph + rx layer), n = 12, 20 (2);
  - control: random circuits n = 6, 8 (2).
- Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
- Arms: QK3 (level 3, seed 0), R1 (release 2026-10-11.1), C39, C40; psf arms with the plain call with backend=.
- Runnable: best ESP of the four >= 0.01; ESP of an output on a failed element = 0. Ratios: geometric means.

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| Y1 | C40 outputs on a failed element, per device | 0 | any |
| Y2 | every test: ESP C40 >= ESP R1, per device (by design of item 69) | all | any lower |
| Y3 | every test: ESP C40 >= ESP C39, per device | all | any lower |
| Y4 | qaoa, runnable, pooled: ESP C40 / R1 | >= 1.10 | < 1.02 |
| Y5 | trotter_commuting, pooled: two-qubit count C40 / R1 (merging) | <= 0.60 | > 0.90 |
| Y6 | trotter_noncommuting + single_noncommuting, runnable, pooled: ESP C40 / R1 | >= 1.05 | < 1.01 |
| Y7 | controls: C40's output = R1's by value, per device | all | any differs |
| Y8 | median compile time of C40 on readable tests (reported, with R1's) | <= 2 s | -- |
| Y9 | rotation_gates, runnable, pooled: ESP C40 / R1 (reported) | -- | -- |

Between the confirmed and refuted bounds: ambiguous.

Proposed rule for taking items 71-72 into the next bundled release (the owner decides): Y1, Y2, Y3, Y7 confirmed on
every device, and at least one of Y4, Y5, Y6 confirmed pooled.
