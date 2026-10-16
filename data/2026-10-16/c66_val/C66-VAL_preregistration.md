# Pre-registration: C66-VAL, the validation of prototype c66 (item 103: item 88's frame in a template by segments) (2026-10-16)

**Status: pre-registration.** Fixed by its SHA-256, given to the owner in the conversation before the run, with the
time in CEST; committed before the run and pushed with the results. Nothing runs on a device.

**What is tested:** [`patches/psf_compile_c66_2026-10-16/psf_compile.py`](../../../patches/psf_compile_c66_2026-10-16/)
(`2026-10-16.c66`) = release 2026-10-16.1 + item 103 (Addendum 511). Only `compile_template`'s frame path (F targets)
changes; the default call and the CZ path are the release's code.

**Where and how** (home, the authors' cloud, 2 CPUs; Qiskit 2.5.2, Aer 0.17.2): `run_c66_val.sh`, from the PSF-Zero root.

1. **Parts R and D:** C63-VAL's script unchanged (`w34/c63_val.py`, `C63_FILE` = c66): the 19 tests; part R on FakeFez
   (T and F; the build input, 1,000 same-sign and 100 free draws), part D on FakeMarrakesh, FakeKingston, FakeTorino
   and the Marrakesh snapshot (the build input and 50 same-sign draws); reference c62. The same random labels as
   C63-VAL and C63A-VAL, so the inputs are theirs.
2. **Default call:** `default_call.py`: c66's `compile_for_hardware` against release 2026-10-16.1's on all 49,590
   inputs and the refusals, bit for bit.
3. **Walk:** `walk_c66.py` (W3's `w3_walk.py`, variant A's file set to c66): C61-VAL part K's 100,000 binds.

## Criteria

| ID | criterion |
|---|---|
| C1 | Native templates (ghz_n12 and the 7 W1 tests), parts R and D: every output identical by value to c62's compilation of the same input. |
| C2 | Item 88's frame (F, chain/graph), parts R and D: every output exact within 1e-6 (where checked), native, free of failed elements, rzz in range, with the same two-qubit count as c62's. |
| C3 | The CZ path (T, chain/graph), parts R and D: every output exact within 1e-6 (where checked), native, free of failed elements. |
| C4 | Shape changes (rebuilt + switched), every cell of parts R and D, same-sign and free draws: 0. A value of exactly 0 changing the shape is known (probability 0 in these draws). |
| C5 | The default call: c66's output equal to release 2026-10-16.1's, bit for bit, on every input and refusal. |
| C6 | The walk: 0 shape changes in 100,000 binds; every 1,000th output exact within 1e-6. |

C1-C3 are scored by the workplace's `c63a_score.py` (copied here unchanged; its A-V2 is C2), C4 from the rows'
shape-change counts, C5 and C6 from their scripts' outputs.

**Reported, not criteria (the owner's decision, 2026-10-16):** the duration over c62's per F cell (median, largest,
share above c62); the bind times (C63-VAL's S1 and S2 measures; this machine is not the workplace's); the walk's bind
time, duration and memory. The authors' expectation from the exploratory measurement (Addendum 511), stated before the
run: no F cell above 1.00 x c62's median duration; bind about 1.5 x option A's.

**Not checked:** exactness above 20 active qubits (tfim_n40_r2), as in C63-VAL.

**Interruption:** a run that stops is started again into a new folder; the first complete one is scored.
