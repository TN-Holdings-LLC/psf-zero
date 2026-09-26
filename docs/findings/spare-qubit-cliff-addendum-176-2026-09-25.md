# Addendum 176 -- Compound round-trip chain results: A1-A5 confirmed, A6 (byte-identical replication) refuted; nothing compounds, and a chained round trip cannot see the symmetric bit-order bug (2026-09-25)

> **Imported into the home series as Addendum 176.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `roundtrip-chain-results-2026-09-25.md`; body below unchanged. Re-computed at home from `roundtrip_chain_2026-09-25.csv` (SHA-256 `b8aa9314...`, as this record states): NEW round trips exact at every step with meaning infidelity at most 3.6e-15; OLD round trips also exact at every step; OLD step-1 meaning infidelity above 1e-3 on 19/19 tapes (minimum 0.919).

**Scored against:** [`roundtrip-chain-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-175-preregistration-2026-09-25.md) (locked
in the Project before the pod run). As that document states, A1-A5 were not
blind predictions: the sandbox dry run had already produced them with the
same converter files. A6, the replication test, was the only open question.
Every verdict below was recomputed in the workplace sandbox from the pod's
raw CSV and matches the script's own scoring.

**Run:** RunPod pod, `python -u roundtrip_compound_chain.py`, numpy 2.5.3,
scipy 1.18.1, Qiskit 2.5.2 (the sandbox dry run: numpy 2.4.4, scipy 1.17.1,
Qiskit 2.5.2). NEW = repository HEAD 2501c7e, OLD = the pre-fix workplace
copy. 3,800 rows. No timing, GPU or backend. **Script integrity:** the
post-run check returned the pre-registered normalized SHA-256
`8aa601aa11041b71d0b57067c5a3ca7812ef2c7d25264ac4c9c5103ecb640e44` (the check
was run after the experiment, not before it).

## 1. Scoring

| Prediction | Verdict | Pod numbers |
|---|---|---|
| A1 NEW round trip fingerprint-exact at every step | CONFIRMED | 1,900 of 1,900 steps; max rt_infid 3.3e-15 |
| A2 NEW meaning_infid <= 1e-12 at every step | CONFIRMED | max 3.6e-15 |
| A3 OLD round trip fingerprint-exact at every step | CONFIRMED | 1,900 of 1,900 steps |
| A4 OLD meaning_infid > 1e-3 at step 1, all 15 F1/F2 tapes | CONFIRMED | 15 of 15; min 0.919 |
| A5 XOR: NEW <= 1e-12 at every step; OLD > 1e-3 at step 1 | CONFIRMED | NEW max 8.9e-16; OLD 4 of 4, min 0.931 |
| **A6 pod CSV byte-identical to the sandbox dry run** | **REFUTED** | SHA-256 `b8aa9314132a73f6b60665c3d5bf14511dc1e7311cbbd8d3116a081072d98a30` (predicted `5bc5b94e...f9c8`) |

Per family on the pod (NEW steps exact / max meaning_infid | OLD steps
exact / min step-1 meaning_infid): F1 1000/1000, 3.6e-15 | 1000/1000, 0.919;
F2 500/500, 2.6e-15 | 500/500, 0.937; F3 XOR 400/400, 8.9e-16 | 400/400,
0.931.

## 2. What the results show

1. **Nothing compounds.** On the pod, as in the sandbox, every chain took a
   single `meaning_infid` value over all 100 steps, for both converters:
   step 1 returns the original tape exactly, so every later step repeats
   step 1. The chain is idempotent; 100 steps carry exactly the error of one.
2. **A round trip, however long, cannot see a symmetric convention error.**
   The pre-fix converter passed the round trip at all 1,900 steps while its
   Qiskit circuits meant something else (average-gate infidelity 0.92-0.94
   against the original). Only the direct meaning check caught it. For the
   converter, "round trip passes" is not evidence of correctness.
3. **The XOR structure is handled correctly by the fixed converter** (with
   CNOT written as a matrix). The 2026-09-28 path does not call the
   converter, so this is not a new check of that path.

## 3. Why A6 failed (locked reading, then the comparison)

Following the reading fixed in the pre-registration, the printed `fp0`
values were compared with the sandbox's:

- **F3 (XOR), 4 tapes: identical.**
- **F1 and F2, 15 tapes: all different.**

A row-by-row comparison of the two CSVs (same 3,800 keys in the same order)
gives:

| Column | Rows that differ |
|---|---|
| converter, family, tape, step, fp_equal, n_ops | 0 |
| fp0 | 3,000 (every F1/F2 row) |
| rt_infid | 2,800 |
| meaning_infid | 1,400, all of them NEW rows; largest difference 2.3e-15 |
| OLD meaning_infid | 0 (identical to all printed digits, every tape) |

**Reading.** OLD's step-1 infidelities (0.919-0.941) match the sandbox to
every printed digit on all 19 tapes. Tapes with different structure (other
qubit pairs, other gates) could not do that, so the F1/F2 tapes have the
same structure on both machines. What differs is the exact bytes of some
random 2-qubit matrices: the fingerprint hashes them bit for bit, and XOR,
whose only matrix is the exact 0/1 CNOT, is unaffected. The remaining
differences (NEW infidelities near 1e-15) are floating-point rounding in the
matrix products, on the order of machine precision.

**Cause not isolated.** The two environments differ in numpy (2.5.3 versus
2.4.4) and scipy (1.18.1 versus 1.17.1) and possibly in the linear-algebra
library underneath. One element checked on the pod,
`random_unitary(4, seed=1).data[0,0]`, is bit-identical to the sandbox's
(`0x1.d7858cab452a0p-4`), so the difference, if in the random unitaries, is
not in every element. Which library and which elements are responsible was
not determined. (An earlier chat remark that the random-unitary generation
was the cause, and a later one suggesting numpy's random stream, were
hypotheses; the second is ruled out by the matching OLD values above.)

**Consequence.** Bit-for-bit reproducibility of this project's outputs holds
within a machine (the long-run audit: 200 identical PSF-Zero outputs) but
should not be expected across machines with different numeric libraries,
even with identical seeds. Structural fields (gate counts, pass/fail
verdicts) reproduced exactly. Future replication gates should compare
structure and tolerance-bounded values, not file hashes, unless the library
versions are pinned. (Stage 1c's R0 failure had a different cause, the
bit-order fix, not this.)

## 4. Files

| File | What it is |
|---|---|
| [`psf-zero/data/roundtrip_chain_2026-09-25.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/roundtrip_chain_2026-09-25.csv) | pod raw data, 3,800 rows (348,466 bytes as received) |
| [`psf-zero/benchmarks/roundtrip_compound_chain.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/roundtrip_compound_chain.py) | the locked script |
| [`roundtrip-chain-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-175-preregistration-2026-09-25.md) | the pre-registration (with the dry run and its `fp0` values) |

Pre-publication grep of the CSV for account names, local paths and host
names: 0 hits.
