# Pre-registration: C63A-VAL, the home part (part D, A-D, A-K) (2026-10-16)

**Status: pre-registration.** Fixed by its SHA-256, given to the owner in the conversation before any of these runs
starts, with the time in CEST. Committed afterwards with the results. Nothing runs on a device.

**What C63A-VAL is:** the validation of option A, c63a (`2026-10-15.c63a`, `TEMPLATE_FRAME_FIXED = True`;
data/2026-10-15/workplace/workplace_2026-10-15c_W1-W3/w34/w3/c63a/psf_compile.py), with C63-VAL's tests, random
draws and measures (workplace note 9, W2). The workplace has only two hours left today, so the work is split (the
owner's decision, 2026-10-16):

- **the workplace:** part R (FakeFez, T and F, with the bind-time criteria), under its own pre-registration;
- **home (this document):** part D, A-D and A-K. None of these measures time.

**The owner's decision on A-V2 (before anything ran):** the home pre-study (FakeFez, F, 11 tests x 20 inputs from
the RNG label "C63A-PRE|...", not C63-VAL's) found c63a's template 1.205 x c62's duration on tfim_n12 and n16, 1.146 x
on tfimP_n12 and n16, 1.067 x on hhI_n16 and n20, at most 1.032 x on the other five, with the same two-qubit count
everywhere; constant over inputs, the build input included. c63's template and c63a's default call are as long as
c62's. A-V2's duration bound (1.05 x) is therefore reported only, not a criterion; options A and B are decided after
the results.

## What runs (home, the authors' cloud; Qiskit 2.5.2, Aer 0.17.2; from the PSF-Zero root)

1. **Part D:** `C63_FILE=<c63a> python <w34>/c63_val.py <out>/D D`, c63_val.py unchanged (FakeMarrakesh,
   FakeKingston, FakeTorino, the Marrakesh snapshot; T and F; 19 tests; the build input and 50 same-sign draws).
2. **A-D:** `python data/2026-10-16/c63a_val_home/a_d.py <out>/a_d.json`: c63a's `compile_for_hardware(qc, target=)`
   against c63's on all 49,590 C63-VAL inputs (parts R and D, regenerated with C63-VAL's own random labels) and the
   refusals (no Target, 19 tests; a three-qubit gate, 5 devices x T and F).
3. **A-K:** `python <w34>/w3/w3_walk.py A <out>/a_k.json` (C61-VAL part K's walk: tfim_n12, FakeFez F, 100,000 binds),
   alone, after 1 and 2.

## Criteria

| ID | criterion |
|---|---|
| A-V1 (part D) | Native templates (ghz_n12 and the 7 W1 tests): every output identical by value (value_sig) to c62's compilation of the same input (C63-VAL's V1). |
| A-V4 | Part D: A-V1's identity for native; exact within 1e-6 (where checked), native, no failed element and rzz in range for chain/graph (C63-VAL's V4). |
| A-D | Every input of A-D: c63a's output equal to c63's, bit for bit (operations, angles as a float's hex, global phase, initial and final layout); every refusal the same exception type. |
| A-K | Shape changes (rebuilt + switched) over the 100,000 binds: 0. A value of exactly 0 changing the shape is known (probability 0 in this walk). |

A-V1 and A-V4 are scored by C63-VAL's rules (c63_score.py's V1 and V4) on the part D rows; if the workplace's
c63a_score.py arrives, it is used as well, and any difference between the two is reported.

**Reported, not criteria:** per F cell of part D, the duration of c63a's output over c62's (median, max); the two-qubit
counts; A-K's bind time and peak memory.

**Interruption:** a run that stops is started again into a new folder; the first complete one is scored.
