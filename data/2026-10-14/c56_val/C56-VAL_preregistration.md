# C56-VAL: pre-registration (fixed 2026-10-14 14:44 CEST, before the run)

**What is tested.** Prototype `2026-10-14.c56` is c55 with item 94 off (dropped, Addendum 488) plus **item 96, the scope
gate** (Addendum 490). With the device's Target, the input's two-qubit interaction graph is tested for an embedding
in the device's working couplers.

- If it embeds, the call is c55's.
- If it does not, the input is handed to Qiskit's level 3 (`generate_preset_pass_manager(3, target, seed 0)`)
  unchanged. If that output uses a failed element, Qiskit runs on the working qubits instead.

Offline only.

**Files fixed by SHA-256:**

    0712599229d7d881061086137be2a1de3b0df2a6d8dcbd5e9898de5d2197e2f7  patches/psf_compile_c56_2026-10-14/psf_compile.py
    dac6fd1c57d5fed29a8287fa5e6ca7adcbba7fcb7f2f04b0c92044bd5d2df0dd  patches/psf_compile_c56_2026-10-14/make_c56.py
    85f9a1d80ea8c2019795e472cffc67a2fd38ab076212d433974853071064ed83  patches/psf_compile_c56_2026-10-14/c56_96_code.py
    84caf8a13aea5cd40c35a2ee076005ae545a591b88093170e1051585e1c09f38  patches/psf_compile_c56_2026-10-14/test_c56.py
    dbb86d7f46ed834144e2a4852939c9d7b81ef2b7a9416dd00d5c6558fa91ca8c  benchmarks/c56_val.py

**Checks before the run.**

- 120 tests passed on c56: item 96's 10 new tests, and c55's, c54's, c53's, c52's, c51's, c49's and c48's suites with
  the gate off.
- a18's 20 tests pass on c56.

**Seen before the run.** The scope reading of C55-VAL's 28 tests on FakeFez and FakeTorino (12 in, 16 out, the same on
both). No compiled output of this run's cells has been seen.

**Run.** `python benchmarks/c56_val.py data/2026-10-14/c56_val/run`.

- **Scope.** C55-VAL's 28 tests on its 5 devices.
- **Arms.** QK3 and QK3F (Qiskit level 3, seed 0); C55 and C55F (item 94 off); C56 and C56F. "F" means the
  fractional Target.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | Every C56 and C56F output exact (<= 20 active qubits, 1e-6), with no refusal. |
| V2 | In scope: C56 / C56F identical to C55 / C55F on every cell. |
| V3 | Out of scope: C56 / C56F identical to Qiskit level 3 (QK3 / QK3F) on every cell, except where Qiskit's output uses a failed element and C56's does not. |
| V4 | The scope is as listed in the script (12 tests in, 16 out) on every device. |
| V5 | C56F gives the same output on a second call (the same cells as C55-VAL). |
| V6 | Every C56F output native, with every rzz angle in range. |
| V7 | Compile time: C56F <= 1.5 x C55F + 2 s on >= 90% of cells. |
| V8 | No C56 / C56F output uses a failed element. |

Reported without a threshold: duration ratios in and out of scope (out of scope, c56 is expected to be longer than c55
offline: that is the price the scope accepts on the evidence of the devices), SCOPE96_STATS and EXACT93_STATS.

**Rule.** If V1-V6 and V8 are confirmed, c56 with a18 is the release candidate. It is released after the owner's
go-ahead; no real-device test is required for it, as the owner decided on 2026-10-14.
