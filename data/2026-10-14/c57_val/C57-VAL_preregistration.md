# C57-VAL: pre-registration (fixed 2026-10-14 15:43 CEST, before the run)

**Why.** C56-VAL (prestatement `7eda6d9b...`, run 14:44-15:39 CEST) confirmed V1-V5, V7 and V8 and **refuted V6**.

- On the fractional Target, 79 of the 80 outputs handed to Qiskit's level 3 carried rzz angles outside IBM's
  executable range 0 < theta <= pi/2. Only a qiskit-ibm-runtime backend object folds them; a Target does not.
- By C56-VAL's rule, c56 is not a release candidate.

**What is tested.** Prototype `2026-10-14.c57` is c56 with item 96 revised: on a Target with rzz, the handed-on output is
folded by item 86's `_fold` (FoldRzzAngle, then single-qubit runs merged). If folding fails, c55's own path is used.
Nothing else changes.

**Files fixed by SHA-256:**

    28942b9f4ac09c5c7dd43c76212a783fdb4231f79c7bf20b5f94912aed6903d1  patches/psf_compile_c57_2026-10-14/psf_compile.py
    d4b563e02d2d564ccdbe9b5b62b7708ee267e644bbce11d4980b538826964fe2  patches/psf_compile_c57_2026-10-14/make_c57.py
    043abdd795de1ebd6eed04eb8fe53cd89e530ecc96bc2b701518c783a17d540b  patches/psf_compile_c57_2026-10-14/test_c57.py
    560ce709bad84823b7583328f70357f8a918de1aedc0ea17da6bd92544e0590a  benchmarks/c57_val.py
    ae5b6660174d805822d32c0b005462c57ac57adc7b76882232f9cfc69946b6f7  data/2026-10-14/c56_val/run/c56_val_rows.jsonl

**Checks before the run.** 122 tests passed on c57: c56's 120 and 2 new ones. One new test checks a handed-on output on
a fractional FakeFez: folded, native, identical to Qiskit level 3 folded by `_fold`, and exact. The other checks that
a plain Target's hand-off is unchanged.

**Seen before the run.** C56-VAL's complete results. No c57 output of this run's cells has been seen.

**Run.** `python benchmarks/c57_val.py data/2026-10-14/c57_val/run`.

- **Scope.** C56-VAL's 28 tests on its 5 devices.
- **Arms.** QK3F, plus its output folded by `_fold`, recorded as QK3F+fold; C57; C57F.
- C56's and C56F's outputs are compared from C56-VAL's rows, fixed above, and not recompiled.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | Every C57 and C57F output exact (<= 20 active qubits, 1e-6), with no refusal. |
| V2 | C57 identical to C56 (C56-VAL's rows) on every cell. |
| V3 | In scope: C57F identical to C56F (C56-VAL's rows). |
| V4 | Out of scope: C57F identical to QK3F+fold, except where C56-VAL recorded Qiskit's output using a failed element and C57F's does not. |
| V5 | C57F gives the same output on a second call. |
| V6 | Every C57F output native, with every rzz angle in range. |
| V7 | No C57 / C57F output uses a failed element. |
| V8 | Compile time: C57F <= 1.5 x C56F's (C56-VAL) + 2 s on >= 90% of cells. |

**Rule.** If V1-V7 are confirmed, c57 with a18 is the release candidate. It is released after the owner's go-ahead.
