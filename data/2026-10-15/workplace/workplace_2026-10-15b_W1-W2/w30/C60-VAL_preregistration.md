# C60-VAL: pre-registration (workplace; fixed 2026-10-15 04:30 CEST, before the run)

**What is tested.** Prototype `2026-10-15.c60` (workplace proposal) is c59 with item 99 added.

- Item 99 applies to an in-scope input (item 96) that neither item 87's chain formula nor item 91's graph formula
  reads. Until now c59 refused these with `OutOfScopeError` ("neither formula reads").
- It builds them by formula as written, in one fixed pass:
  1. item 91's placement (the first VF2 embedding);
  2. commuting diagonal gates ordered by item 91's edge colouring;
  3. every two-qubit gate in a fixed native form, each multiplied out and checked;
  4. single-qubit runs merged in closed form (ZYZ), written by PSF-Zero itself.
- No pass manager runs, nothing is compared, nothing is simulated. Item 89 is not applied.
- Inputs that items 87 and 91 build are untouched.

**Files fixed by SHA-256:**

    e6aaef9a1d5655c7f9bec7119c59675869a19b5bf028262f6f1db8220bf93433  c60/psf_compile.py (built by make_c60.py from c59)
    d593b3a3830d815f95ff3e049a81ad40dca98dac15a3f57d6b598bc6e91f6fcd  make_c60.py
    37d460b6963687d555fc40b0a7cf331b4d3c150e67c7a6dd4ba4f0c6fc1afa71  c60_99_code.py
    6fe03993268326fc3daa53a75361c767d09853da13c0c39c3bf657686901307b  c60_tests.py
    d60b02c2d3e48236d13d2ddb5d9ce6fc35b41e1440daaeb11acc94389feccc6d  c60_val.py
    c7e455ea6b4a4048956c5a3a13354328a8c94bd84ef6652ac75bd9c13be77d7a  w4_orient.py (W4's tests, unchanged)
    68df2335aac3b326f1ae6011d1c0a81d854e54d6360fd38314a96e5782ec6258  w4_order.py (W4, unchanged)
    d02006f4d6217fdafac702f5a3c69032add4a6c891c843c4cd7ad850ca7b546d  patches/psf_compile_c59_2026-10-15/psf_compile.py
    23d4aa4d1cdea7729d323c82d63386f43ccb302fc28ffe50f9ae58363459d0f8  psf_compile.py (release 2026-10-14.1 = c57)

**Seen before fixing.**

- c60 on FakeFez, T and F, for all 21 tests of part A: every output was built, exact (largest difference 7e-16),
  native, with rzz in range, at 16-140 ms per call. Durations of c60 only.
- In the script's smoke run, on FakeFez only: chain_cz_n12 and chain_cz_n16. There C57 was 1.62 / 1.57 / 1.96 /
  1.93 µs and QK3 1.72 / 1.57 / 2.08 / 1.93 µs (T, F, T, F), against c60's 0.864 / 0.768 / 0.864 / 0.768. On that
  run's four part B cells, c60 was identical to c59 or refused with it (and was new on chain_cz_n12).
- W4's c57 numbers for chain_cz (2026-10-15a). No other c57 or Qiskit output of this run's cells has been seen.

**Run.** From the repository root (bundle o):

    python <dir>/c60_val.py <out dir>

**Part A: item 99's 21 tests.** These are on the 5 devices of C57-VAL (FakeFez, FakeMarrakesh, FakeKingston,
FakeTorino, ibm_marrakesh's snapshot), each without fractional gates (T) and with them (F): 210 cells. The tests:

- W4's CZ-chain ansatz (n = 12, 16);
- a CZ ring (n = 12);
- CZ heavy-hex patches on C57-VAL's edges (n = 16, 20);
- a CX chain, cx(q, q+1) per layer (n = 12, 16);
- Qiskit's `real_amplitudes` and `efficient_su2`, entanglement "linear", reps 2 and 4, n = 8, 12, 16, with random
  bound parameters, expanded to ry, rz, cx before input;
- note 6's ring classifier on 12 qubits (CZ ring; L = 4 and 8).

Arms, all with seed 0 and C57-VAL's call:

- **C60**, called twice;
- **C57**, the release's default call;
- **QK3**, Qiskit level 3 on the Target. On F its output is folded by c57's `_fold`, as c57 hands it on.

Durations are ASAP on the arm's own Target, measurements left out.

**Part B: regression.** C57-VAL's 28 tests and W4's 10 gate-by-gate tests, as written, through c59 and c60 on the
same 5 devices, T and F: 380 cells.

**Part C (reported only).** W4's 10 tests on FakeFez, T and F, through c60:

- in three orientations (as written, reversed, random);
- with each run of commuting ZZ-type gates written in reverse order.

The spread (max/min) and the number of distinct outputs are reported.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | Part A: every C60 output (210 cells) is built (no refusal), exact within 1e-6 (largest outcome-probability difference; every test has at most 20 active qubits, so a cell that cannot be checked fails), native, free of failed elements, and every rzz angle is in 0 < theta <= pi/2. |
| V2 | Part A: C60's second call gives the same output (signature) in every cell. |
| V3 | Part A, durations: the geometric mean of C60 / C57 over the cells where C57 returns an output is <= 1.00, and no cell has C60 / C57 > 1.05. |
| V4 | Part B: wherever c59 returns an output, c60's is identical (signature). Wherever c59 refuses because the call is out of scope (an operation on more than two qubits, or no embedding), c60 refuses too. |
| V5 | Part B: wherever c59 refuses with "neither formula reads" and c60 builds, c60's output is exact within 1e-6 (where at most 20 qubits are active), native and free of failed elements. |

**Reported, not predicted.**

- C60 / QK3 durations (geometric mean, largest).
- Two-qubit counts of all three arms.
- Compile times (C60, C57, QK3; c59 and c60 in part B).
- Part C's spreads.

**Rule.** If V1-V5 are all confirmed, c60 (item 99) is proposed to home as the next prototype. If any one is refuted,
the result is reported and item 99 is revised before it is proposed. The criteria are not changed after the run.
