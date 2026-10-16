# C63A-VAL, part R (workplace): pre-registration (fixed before the run, at the CEST time in PREREG_SHA256_C63A_R.txt)

**Task.** Note 9 W2, as decided by home and the owner on 2026-10-16. The workplace runs part R only. Home runs part D,
A-D and A-K under its own pre-registration.

**What is tested.** Prototype `2026-10-15.c63a`: option A, item 88's frame fixed at build in the compile template. It
is tested with C63-VAL's tests, draws and measures, against c62's default call.

## Files fixed by SHA-256

    9d0fecbd09c4745d5384324fc66fea644deca95b2b07a8e06df74aad5b209492  w34/w3/c63a/psf_compile.py (the 2026-10-15c delivery)
    28de2c381a8e9e748eb35a4291d374a8826e56749c923999adc9f86f31001c8a  w34/c62/psf_compile.py (the reference)
    ee9e4cc5bb98830b4ba7e7df44d2b6814931a2a05bb7f38d244f7efb7a4c783e  w34/c63_val.py (unchanged; C63_FILE points it to c63a)
    c65c7df05a7e1d78bf6e01d9eb82211678cebc2678d28d28925b2259b00403f5  run_r.sh (one segment per test)

The scorer `c63a_score.py` is written after fixing. It applies exactly the criteria below, and its SHA-256 is reported.

## Run

From the repository root (bundle q), with nothing else on the machine: `sh run_r.sh <w34>`.

- **Command per segment:** `OMP_NUM_THREADS=1 C63_FILE=w34/w3/c63a/psf_compile.py C63_ONLY=<test> python w34/c63_val.py run_r/R_<test> R`.
- **Tests and order:** C63-VAL's 19 tests, in C63-VAL's order, on FakeFez, T and F.
- **Inputs per cell:** the build input, 1,000 same-sign draws and 100 free draws, with C63-VAL's seeds.
- **Unfinished work:** the workplace delivers what is finished before leaving. An unfinished segment is reported as
  not run; no verdict is drawn from a partial set.

## Seen before fixing: home's probe

- **Set-up:** inputs with a separate seed ("C63A-PRE|..."), FakeFez F, 11 tests x 20 inputs.
- **c63a's template against c62 (circuit duration):**

  | tests | ratio |
  |---|---|
  | tfim_n12, tfim_n16 | 1.205 |
  | tfimP_n12, tfimP_n16 | 1.146 |
  | hhI_n16, hhI_n20 | 1.067 |
  | the other 5 | <= 1.032 |

- Two-qubit counts were equal everywhere.
- The ratio does not depend on the input, and the build input gives the same ratio.
- c63's template and c63a's default call have c62's durations.
- W3's "largest 1.044" came from a different measurement and does not agree with this one.
- **Owner's decision:** the duration condition of A-V2 (<= 1.05 x c62) is **not** a criterion. It is reported only.
  Equal two-qubit counts remain a criterion. Option A or B is decided later, from the results.

## Criteria (part R)

| ID | criterion |
|---|---|
| A-V1 | Native templates (ghz_n12 and W1's 7 tests): every output identical by value (value_sig) to c62's compilation of the same input. |
| A-V2 | Item 88 frame (F, chain/graph): every output exact within 1e-6 (where checked), native, free of failed elements, rzz in range, with **two-qubit count equal to c62's**. |
| A-V3 | CZ path (T, chain/graph): every output exact within 1e-6 (where checked), native, free of failed elements. |
| A-S1 | Bind median, interleaved (C63-VAL's S1 measure, same-sign draws without a shape change): <= 2 ms in every cell with n <= 20, and <= 5 ms for tfim_n40_r2. |
| A-S2 | The same, bound back to back (C63-VAL's S2 measure). |

- **Exactness** is checked as in C63-VAL: on every input at <= 16 active qubits, on every 10th input and the build input at
  17-20, and not at 40.
- **Expected from C63-VAL** (stated now; not a criterion):
  - The T path and the native path of c63a are c63's code, so A-V1, A-V3 and the T cells' bind times should repeat
    C63-VAL's.
  - S1 missed there in heis_n12_r2 T, hhI_n16 T, hhI_n20 T and F, and hhtree_heis_n16 T and F. S2 missed in
    hhtree_heis_n16 T.

**Reported, not criteria:**

- per F cell, duration against c62: median ratio, largest ratio, and the share of outputs above c62;
- shape changes per cell (rebuilt + switched), same-sign and free draws separately;
- values_of, own assign_parameters, Qiskit's alternative, build and compile times (as C63-VAL).

**Rule.** The criteria are not changed after the run. Home combines this part with its part D.
