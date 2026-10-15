# C63-VAL: pre-registration (workplace; fixed before the run, at the CEST time in PREREG_SHA256_C63.txt)

**What is tested.** Prototype `2026-10-15.c63` (workplace proposal) is c62 with item 100, the compile template, with
a faster binding (W2 of note 8). c62's default call is unchanged.

`compile_template(qc, target)`, `CompileTemplate.values_of`, `.bind_values`, `.bind` and `.signature` are as in c61.
What changes is how a bind is computed:

- **Planned binding.** Each output shape keeps its emission plan: for every single-qubit run, the positions of its
  factors, its class and its slots. A bind multiplies each run's factors and checks its class against the plan; it
  does not re-emit. If a class differs, the shape changed: it is built anew or taken from the cache, as in c61.
- **Constant runs baked in.** A run whose factors are all fixed (fixed gates, Hadamards, local Cliffords) has its
  angles written into the circuit as numbers. Only the other angles remain parameters.
- **One numpy batch for items 87/91.** On the chain and graph templates, all runs' products, ZYZ angles and classes
  are computed in one numpy batch per bind (the note's proposal). Item 99's runs (the native kind) keep CPython's
  arithmetic, because T1 asks for identity with c62 bit for bit, and numpy's complex product is not bitwise
  CPython's.
- **Pinned runs at Weyl blocks.** On the CZ path, a block of several terms with single-qubit operations between them
  is emitted by Qiskit's Weyl decomposition in closed form, as in c61. A run that contains a factor of such a block
  is always emitted in the full form (rz sx rz sx rz). The decomposition's discrete branches therefore no longer change
  the shape. In c61, hhtree_heis_n16 T changed shape on nearly every draw.
- **One batched check.** The Weyl decompositions are checked in one numpy batch per bind, to item 99's block tolerance.
  Blocks with c = 0 keep their own per-block check.

`TEMPLATE_BATCH = False` restores CPython's per-run arithmetic on every template.

**Files fixed by SHA-256:**

    9017689102b0369d1dd730b7c02b5c9bf7198b5cd63d1a30484bb4a2331d4087  c63/psf_compile.py (make_c63.py on c62)
    425c9cf73fdde3874f93fde3dc364632ba5443fc35310a091c6ccb09356d92a4  make_c63.py
    c4aade5bb30a2914374989f851f8f9891e729a7f6306d121a9e3a71fd5a09c81  c63_100_code.py
    ee9e4cc5bb98830b4ba7e7df44d2b6814931a2a05bb7f38d244f7efb7a4c783e  c63_val.py
    13771bd67ba5f74fa20efb13d37e936211ec096d93460e87fbc98df8e0fe1d1f  c63/test_c63.py (8 passed before fixing)
    28de2c381a8e9e748eb35a4291d374a8826e56749c923999adc9f86f31001c8a  c62/psf_compile.py (C62-VAL's)
    6fe03993268326fc3daa53a75361c767d09853da13c0c39c3bf657686901307b  c60_tests.py (bundle q)

**The reference is c62's default call.** It is the base for W2. C62-VAL showed c62 identical to c60 in every F cell
and every item-99 cell. On the T cells of items 87/91, c62 emits its own CZ path (item 101).

**Correctness conditions: C61-VAL's T1, T2 and T6, as note 8 asks.**

- C61-VAL's T3 (CZ path: duration and two-qubit count <= the reference's) is **not** carried over as a criterion.
- Reason: on Heisenberg blocks, the template keeps c61's fast closed form for XX + YY + ZZ and pins Weyl-block runs to
  the full form. c62's default call uses item 101's Weyl form. So the template is not expected to match c62 in
  duration there; the smoke run showed it longer.
- Duration and two-qubit count against c62 on the T cells are **reported per cell**.
- The CZ path is still held to the correctness part of T3: exact, native, no failed element (V3).

**Seen before fixing.**

- **Isolated bind-time probe** (FakeFez, same-sign draws, one process, c61 against c63, all 38 cells). Median c63 bind
  in ms:
  - every cell <= 2 ms except hhtree_heis_n16 T (3.5) and hhI_n20 T (2.04);
  - tfim_n40_r2: T 1.84, F 1.11;
  - repeated runs of the probe varied by 20-50 % on this machine (2 CPUs).
- **Smoke run of this script, part R** (FakeFez, 3 same-sign + 2 free draws, all 38 cells):
  - every output checked was exact;
  - every native output was identical to c62's;
  - every F output had duration and two-qubit count equal to c62's;
  - Ising T cells: duration and two-qubit count equal to c62's in every input;
  - Heisenberg T cells: two-qubit count equal everywhere; duration equal in 1 of 6 inputs for heis_n12_r2 (longer in
    the other 5) and in 0 of 6 for hhtree_heis_n16 (longer in all).
- **Smoke run, parts R and D** (ghz_n12 and hhtree_heis_n16, 20 draws): all exact; native all identical.
- The interleaved bind medians in the smoke runs (3-20 draws) ranged from 0.2 to 11 ms. They were noisy and higher
  than the isolated probe's. In C61-VAL, the interleaved median was 1.3-1.7 x the isolated one.

**Run.** From the repository root (bundle q), with `OMP_NUM_THREADS=1`. Part R first, alone; then part D:

    python <dir>/c63_val.py <out>/R R
    python <dir>/c63_val.py <out>/D D

**Tests.**

- C58-VAL's 12 in-scope tests: tfim_n12, tfim_n16, tfimP_n12, tfimP_n16, tfim_n20_r3, heis_n12_r2, tfim_n40_r2, hhI_n16,
  hhI_n20, ring12_tfim, hhtree_heis_n16, ghz_n12 (item 99's).
- W1's 7 CZ / CX ansatz tests (item 99): chain_cz_n12, chain_cz_n16, ring_cz_n12, hh_cz_n16, hh_cz_n20, cx_chain_n12,
  cx_chain_n16.
- 19 tests, T and F: 38 cells per device.

**Draws.**

- **Part R** (FakeFez): the build input, 1,000 same-sign draws (every angle and coefficient times a factor uniform in
  [0.5, 1.5]) and 100 free draws (angles uniform in [-pi, pi], coefficients in [-1.5, 1.5]).
- **Part D** (FakeMarrakesh, FakeKingston, FakeTorino, the Marrakesh snapshot): the build input and 50 same-sign draws.
- **Exactness** is checked on every input at <= 16 active qubits, and on every 10th input and the build input at
  17-20. It is not checked at 40 qubits (tfim_n40_r2).

**Predictions.**

| ID | prediction |
|---|---|
| V1 (T1) | Native templates (ghz_n12 and the 7 W1 tests), parts R and D: every output identical by value (value_sig) to c62's compilation of the same input. |
| V2 (T2) | Item 88 frame (F, chain/graph), part R: every output exact within 1e-6 (where checked), native, free of failed elements, rzz in range, with duration and two-qubit count equal to c62's (duration to 1e-9 us). |
| V3 | CZ path (T, chain/graph), part R: every output exact within 1e-6 (where checked), native, free of failed elements. |
| V4 (T6) | Part D: V1's identity for native; exact within 1e-6 (where checked), native, no failed element and rzz in range for chain/graph. |
| S1 | Bind time, C61-VAL's T4 measure (each bind timed in the loop, between c62's compilation and the exactness check; same-sign draws without a shape change), part R: median <= 2 ms in every cell with n <= 20, and <= 5 ms for tfim_n40_r2 T and F. |
| S2 | The same, with the same-sign values bound back to back after the loop (no compilation or simulation in between). |

**Expected from the probe** (stated now; not a criterion):

- hhtree_heis_n16 T will miss S1 and S2. Its 11 general Weyl blocks need a Weyl decomposition on every bind, and
  their items are generated in Python.
- hhI_n20 T is borderline.
- S1 is at risk in more cells than S2, through the interleaving.

**Reported, not predicted.**

- Per cell:
  - the values_of median;
  - the bind's own `assign_parameters` median (fresh slot values, 200 calls);
  - Qiskit's alternative for gate-level inputs (level 3 on a parametrised copy, then assign_parameters; as C61-VAL);
  - rebuilt and switched counts;
  - build time and c62 compile time (medians of 3).
- On the T cells, duration and two-qubit count against c62. On the non-native cells, the number of outputs identical to
  c62's.

**Rule.**

- Correctness is never traded: V1-V4 must hold. A refuted V1-V4 means the template is revised.
- S1 and S2 are speed targets. A miss is reported with its cause.
- The criteria are not changed after the run.
