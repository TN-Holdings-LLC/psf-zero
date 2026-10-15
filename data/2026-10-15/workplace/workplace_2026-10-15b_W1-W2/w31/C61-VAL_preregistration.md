# C61-VAL: pre-registration (workplace; fixed 2026-10-15 05:50 CEST, before the run)

**What is tested.** Prototype `2026-10-15.c61` (workplace proposal) is c60 with item 100, the compile template.

- `compile_template(qc, target, seed_transpiler=0)` returns a `CompileTemplate` with `.values_of(qc)`,
  `.bind_values(values)`, `.bind(qc)` and `.signature`.
- It builds an in-scope circuit once, by the formula path c60 takes (item 87, 91 or 99), into a parametrised circuit.
  New values are bound in closed form only: no pass manager, no simulation.
- A shape signature covers every value-dependent branch. When it changes, the new shape's circuit is built (counted
  "rebuilt"), or taken from the template's cache of shapes (counted "switched"). c60's default call is unchanged.

**Files fixed by SHA-256:**

    254cfabc5924118a5c54e159fa2284202d21b87894202f447333de07f06b3728  c61/psf_compile.py (make_c61.py on c60)
    4878a6f55af56305a3b1cb33acc2b29f50141f4f1aefd0d5a990f582f95ba156  make_c61.py
    f66f5917d67c400c444066d7c32d4c657e4905d5b347b8e244784d83daa74182  c61_100_code.py
    71ae6d02cfcf06bc2b0f00610218074d03037df9857b54ee5bc8e00a74291001  c61_val.py
    872fe88934b2c024d82d22ad4388f0d7d385cd0dfbfb709097da6db43a490b2a  c61/test_c61.py (8 passed before fixing)
    e6aaef9a1d5655c7f9bec7119c59675869a19b5bf028262f6f1db8220bf93433  c60/psf_compile.py (C60-VAL's)
    6fe03993268326fc3daa53a75361c767d09853da13c0c39c3bf657686901307b  c60_tests.py

**Where the note's draft is changed, and why** (decided before the run).

1. **Items 87/91 cannot be reproduced value for value.** Their last step is Qiskit's: Optimize1qGatesDecomposition on
   the frame path (F), or the level-3 preset on the fixed layout (T). A template that may not run a pass manager emits
   them itself:
   - **F:** the same logical circuit, the same RemoveDiagonalGatesBeforeMeasure positions and the same Pauli X frame;
     single-qubit runs merged by PSF-Zero. Condition: exact, native, and **duration and two-qubit count equal to
     c60's**. This is the note's fallback.
   - **T:** every two-qubit term as a local Clifford around RZZ. Consecutive terms on one pair form one block, closed
     form when it is one term or exactly XX + YY + ZZ, else Qiskit's Weyl decomposition with closed-form circuits.
     Diagonal terms directly before the measurements are left out.
     - In the smoke runs this was mostly shorter than c60, so "equal" cannot hold. The condition is: exact, native,
       **two-qubit count <= c60's and duration <= c60's**.
     - This is a deviation from the note's "the same as c60". If home wants equality, this emission must be
       revisited.
2. **"Random angles with the same signature."** Each draw multiplies every rotation angle and every Pauli coefficient
   of the input by a factor uniform in [0.5, 1.5], so signs are kept. Item 88's frame changes shape with signs. Draws
   whose shape still changes are counted and still checked. In addition, 100 free draws per test (every angle uniform
   in [-pi, pi], every coefficient uniform in [-1.5, 1.5]) test the rebuild path. Every output, from either kind of
   draw, must meet its condition.
3. **Exactness at 17-20 active qubits** is checked on every 10th draw and on the build input, because a 20-qubit state
   vector costs about 1.3 s per check. At 16 or fewer, every draw is checked. tfim_n40_r2 (40 qubits) cannot be
   checked; there only the duration, two-qubit and native conditions apply.

**Seen before fixing.** Smoke runs with 3-8 draws per test on FakeFez, and 1 draw on the other devices:

- every output met its exactness, native and identity conditions;
- heis_n12_r2 T was 0.3% longer than c60 in some draws (the XX + YY + ZZ closed form);
- hhtree_heis_n16 T changed shape on most same-sign draws;
- bind times of 0.3-4.8 ms (machine shared with other runs);
- the 100k loop at 1,000-2,000 laps rebuilt often before the cache was added.

**Run.** From the repository root (bundle o), in two processes for part R plus one for D and K:

    C61_ONLY=<first 10 tests> python <dir>/c61_val.py <out>/r1 R
    C61_ONLY=<last 9 tests>   python <dir>/c61_val.py <out>/r2 R
    python <dir>/c61_val.py <out>/dk D K

**Tests.**

- C58-VAL's 12 in-scope tests: tfim_n12, tfim_n16, tfimP_n12, tfimP_n16, tfim_n20_r3, heis_n12_r2, tfim_n40_r2,
  hhI_n16, hhI_n20, ring12_tfim, hhtree_heis_n16, ghz_n12. ghz_n12 is item 99's in c60.
- W1's 7 CZ / CX ansatz tests: chain_cz_n12, chain_cz_n16, ring_cz_n12, hh_cz_n16, hh_cz_n20, cx_chain_n12,
  cx_chain_n16.

**Predictions** (part R: FakeFez, T and F, 1000 same-sign draws + 100 free draws + the build input per test).

| ID | prediction |
|---|---|
| T1 | Item 99 (native) templates: every output identical by value (value_sig) to c60's compilation of the same input. |
| T2 | Item 88 frame (F, chain/graph) templates: every output exact within 1e-6 (where checked), native, free of failed elements, rzz in range, with duration and two-qubit count equal to c60's (duration to 1e-9 us). |
| T3 | CZ path (T, chain/graph) templates: every output exact within 1e-6 (where checked), native, free of failed elements, two-qubit count <= c60's, duration <= c60's. |
| T4 | Bind time: the median of `bind_values` over the same-sign draws without a shape change is <= 2 ms for every test with n <= 20, and <= 5 ms for tfim_n40_r2. |
| T5 | Build time (median of 3) <= 2 x c60's compile time (median of 3), every test and target. |
| T6 | Part D (FakeMarrakesh, FakeKingston, FakeTorino, snapshot; T and F; 50 same-sign draws + the build input): correctness only. T1 identity for native; exact, native, no failed element, rzz in range for chain/graph. |
| T7 | Part K (tfim_n12, FakeFez, T and F, 100,000 binds on a random walk, steps N(0, 0.05) on every value): every 1000th output exact within 1e-6 against the reader's terms. Peak RSS at lap 100,000 at most 20 MB above that at lap 10,000. |

**Reported, not predicted.** For each test: `values_of` time; Qiskit's alternative for gate-level inputs (level 3 on a
parametrised copy, then assign_parameters); counts of rebuilt and switched; the 100k loop's median, 99th percentile
and largest bind time.

**Rule.** The result is reported per prediction and per test. Item 100 is proposed as it stands only if T1-T3 and
T6-T7 are confirmed. Correctness is never traded: a refuted T1-T3, T6 or T7 means the template is revised. T4 and T5
are speed targets; a miss is reported with its cause. The criteria are not changed after the run.
