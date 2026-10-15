# C62-VAL: pre-registration (workplace; fixed 2026-10-15 08:25 CEST, before the run)

**What is tested.** Prototype `2026-10-15.c62` (workplace proposal) is c60 with item 101. Items 87/91's CZ path (no
item 88 frame, i.e. no rzz and rx in the Target) no longer goes to Qiskit's level-3 preset on the fixed layout.
`_cz101_emit` emits it, by block kind; nothing is compared:

- one two-qubit term (a single interaction) as a local Clifford around RZZ, in closed form;
- several terms on one pair (with the single-qubit operations between them) by Qiskit's Weyl decomposition with
  closed-form circuits. This is the form that matched c60 in C61-VAL's probe.

Diagonal operations directly before the measurements are left out, and runs are merged in closed form (theta = pi as
one x). Placement, order and the frame path are items 87/91's. `CZ_OWN_EMIT = False` restores c60.

**Files fixed by SHA-256:**

    28de2c381a8e9e748eb35a4291d374a8826e56749c923999adc9f86f31001c8a  c62/psf_compile.py (make_c62.py on c60)
    21d03e1e6bee5639e2c282200e3b7b560b8e25980507be2d507eb5202e77bad7  make_c62.py
    22bc7792cc8e262a394eccaf9199422ddf18d1fd98ba6be1072e3fd59c89bb76  c62_101_code.py
    8b13cdbbc34faed9ad8ac9d7c72c5ea8d8eccd5de597c5f7d88014914eb9199e  c62_val.py
    e6aaef9a1d5655c7f9bec7119c59675869a19b5bf028262f6f1db8220bf93433  c60 (bundle q, the 2026-10-15b delivery)
    6fe03993268326fc3daa53a75361c767d09853da13c0c39c3bf657686901307b  c60_tests.py (same)

**Seen before fixing.**

- The FakeFez probe: the 12 in-scope tests, T and F.
- A smoke run of this script, stopped during FakeKingston. It covered FakeFez and FakeMarrakesh completely, and
  FakeKingston up to tfimP_n16.
- What those showed:
  - Ising T: 0.75-0.86 x c60;
  - heis and hhtree_heis T: 1.000;
  - every F cell and every item-99 cell: identical to c60;
  - every output exact.
- FakeTorino, the snapshot, and the rest of FakeKingston have not been seen.

**Run.** `python <dir>/c62_val.py <out dir>` from the repository root (bundle q).

- Tests: C57-VAL's 28 (its 12 in-scope tests among them; the others are refused) and C60-VAL's 21.
- Devices: C57-VAL's 5, each T and F: 490 cells.
- Arms (seed 0, C57-VAL's call): C62 (called twice) and C60.

**Groups (fixed now).**

- **Ising** (every block a single term): tfim_n12, tfim_n16, tfimP_n12, tfimP_n16, tfim_n20_r3, tfim_n40_r2,
  hhI_n16, hhI_n20, ring12_tfim.
- **Heisenberg** (XX + YY + ZZ blocks): heis_n12_r2, hhtree_heis_n16.
- **Item 99:** ghz_n12 and C60-VAL's 21.
- Item 101 acts only on the T cells of the Ising and Heisenberg groups. Their F cells take item 88's frame, which is
  unchanged.

**Predictions.**

| ID | prediction |
|---|---|
| P1 | Every C62 output is exact within 1e-6 (where at most 20 qubits are active), native, rzz in range, free of failed elements. |
| P2 | Ising, T cells (45): every cell's duration <= c60's, and the geometric mean of C62 / C60 <= 0.85. |
| P3 | Heisenberg, T cells (10): every cell's duration <= 1.01 x c60's. |
| P4 | C62 identical to C60 (value_sig) in every F cell and every item-99 cell. C62 refuses exactly where C60 refuses. |
| P5 | C62's second call is identical in every cell where it returns an output. |

**Reported, not predicted.** Two-qubit counts against c60; compile times.

**Rule.** If P1-P5 are confirmed, c62 is proposed as the base for W2 and W3 and for home. The criteria are not changed
after the run.
