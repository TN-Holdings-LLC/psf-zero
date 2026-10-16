# Workplace delivery 2026-10-16c: C63A-VAL part R (note 9 W2, split with home)

Offline only: no IBM connection, no commit, no push. Check the delivery with `python check_intake.py`.

- **Pre-registration:** `C63A-VAL-R_preregistration.md`, fixed 2026-10-16 08:20:11 CEST (SHA-256 `bf184f28...`). It
  includes home's probe and the owner's decision: A-V2's duration ratio is reported only.
- **Run:** 08:20:11-09:55:12 CEST, `run_r.sh`, one segment per test. All 19 tests finished; no segment was restarted.
- **Inputs:** `w34/c63_val.py`, unchanged, with `C63_FILE` pointing to c63a (SHA-256 `9d0fecbd...`, checked in every
  head file); the reference is c62.
- **Scoring:** `python c63a_score.py run_r [<part D dirs>]`. The result is in `run_r/c63a_score.txt`.

| ID | result (part R, FakeFez, 38 cells, 41,838 inputs) |
|---|---|
| A-V1 | CONFIRMED: 17,616 native outputs identical to c62 |
| A-V2 | CONFIRMED: 12,111 F outputs exact (largest difference 2.4e-15; 3,081 not checked: 17-40 qubits off the schedule), native, no failed element, rzz in range, two-qubit count equal to c62 in every output |
| A-V3 | CONFIRMED: 12,111 T outputs exact (largest difference 8.8e-15), native, no failed element |
| A-S1 | REFUTED: 34 of 38 (hhtree_heis_n16 F 3.81 and T 3.31, heis_n12_r2 F 2.96, hhI_n20 F 2.21 ms) |
| A-S2 | REFUTED: 36 of 38 (hhtree_heis_n16 F 2.82 and T 2.24 ms) |

## Reported, not criteria

**Shape changes: 0 in every cell, the 100 free draws included.** In C63-VAL (c63), every F cell changed shape 101
times.

**F durations against c62** (median ratio, with the largest in brackets):

| tests | ratio |
|---|---|
| tfim_n12, tfim_n16 | 1.205 |
| tfimP_n12, tfimP_n16 | 1.146 |
| hhI_n16 | 1.067 (1.164) |
| hhI_n20 | 1.067 (1.143) |
| tfim_n20_r3 | 1.032 (1.102) |
| hhtree_heis_n16 | 1.004 |
| heis_n12_r2 | 1.002 (1.005) |
| tfim_n40_r2 | 1.000 (1.044) |
| ring12_tfim | 1.000 (1.078) |

- These agree with home's probe.
- T cells and native cells are c63's code, with the same durations as in C63-VAL.

**F binds in the Heisenberg cells are slower than with c63's frame.** Option A's pinned items are new tuples on every
bind, so fewer runs are baked as constants:

| cell | c63a back to back (C63-VAL's c63) | c63a interleaved (c63) |
|---|---|---|
| heis_n12_r2 F | 1.67 ms (0.61) | 2.96 ms (1.24) |
| hhtree_heis_n16 F | 2.82 ms (1.14) | 3.81 ms (2.05) |

The other F cells are within 0.1-0.6 ms of c63.

**Not done here** (home's part): part D, A-D, A-K.
