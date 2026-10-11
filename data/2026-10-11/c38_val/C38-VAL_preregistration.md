# C38-VAL -- fixed before the run (local; to be recorded in an Addendum with the results)

Written 2026-10-11 07:31 CEST, before the full run; nothing of the full run had been seen
(a smoke run of 4 jobs on one MaxCut circuit, n = 10, checked that the script runs).

- Candidate: psf_compile 2026-10-11.c38 = release 2026-10-11.1 + changelog item 68 (sha256 99bff3a5885c12c735bd09b31abe8404cd4472df026db0b92c7087003de10720),
  built by make_c38.py; test_c38.py (sha256 a62fadea66743f7b7e66bcbcb3bb62e8fef5502c5e7c9e5f695179bc0a977f07): 6 tests passed.
- Script: c38_val.py (sha256 26c66277c7400c8c9172a115e0432ab7e7dcf14575b0d5a0cbd9a99088544233).
- Rule (fixed from STEINER6's development data only, Addendum 444): STEINER6 is used only for one Pauli evolution
  whose terms all pairwise commute; every other input goes by the release's path.
- Circuits never used in STEINER6's development: Benchpress's HamLib instances outside BP-FINAL's 67 that fit the
  device, and generated circuits from fixed seeds (MaxCut-type and 3-SAT-type, commuting; transverse-field Ising and
  Heisenberg rings, not commuting).
- Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston (the last two never used by
  STEINER6). Arms: QK3 (Qiskit level 3, seed 0), REL (release, plain call with backend=), C38 (same call).
- Runnable: best ESP of the three >= 0.01; ESP of an output on a failed element (error >= 0.5) = 0.

| ID | prediction, scored per device | confirmed | refuted |
|---|---|---|---|
| V1 | C38 outputs on a failed element | 0 | any |
| V2 | non-commuting inputs: C38's output = REL's, by value | all | any differs |
| V3 | runnable commuting tests: ESP C38 / REL (geometric mean) | >= 1.10 | < 1.00 |
| V4 | runnable commuting tests: ESP C38 / QK3 | >= 1.05 | < 1.00 |
| V5 | all commuting tests: two-qubit count C38 / REL (geometric mean) | <= 0.95 | > 1.00 |
| V6 | compile time C38 / REL on commuting tests (reported, does not decide) | <= 20 | -- |

Between the thresholds: ambiguous. Proposed release rule (the owner decides): V1, V2 and V3 not refuted on any
device, and V3 confirmed on the six devices pooled.
