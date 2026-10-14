# C49-VAL: pre-registration (fixed 2026-10-14 02:16 CEST, before the run)

**What is tested.** Prototype `2026-10-14.c49` = c48 plus two items.

- **Item 84.** Every choice between two compiled circuits uses the as-soon-as-possible duration on the device's
  reported gate durations, then the two-qubit count. It no longer uses the summed -ln(1 - error).
  - This covers items 69, 74, 80 and 82, and STEINER6's choice among its constructions.
  - Failed elements still exclude.
- **Item 85.** With the device given, its failed qubits are cut out of the device before compiling.

This is offline only; nothing runs on a real device. Whether shorter circuits run better on hardware is a separate,
pre-registered hardware test (TIME1).

**Files fixed by SHA-256.**

| file | SHA-256 |
|---|---|
| `patches/psf_compile_c49_2026-10-14/psf_compile.py` | `5544d7c9f10d615c041eaf273af311573151a2548690b2192d866509bdaaa8d0` |
| `patches/psf_compile_c49_2026-10-14/make_c49.py` | `92223961ff2fa44ffa8c81755bd6e2ae78c1af73126775a6a70f903aaa63351e` |
| `patches/psf_compile_c49_2026-10-14/c49_code.py` | `a3c6b919c08d0ebc798a76cdf3538b9913fab7757d1d01a86a917c29bdc38e90` |
| `patches/psf_compile_c49_2026-10-14/test_c49.py` | `d20bf4d49b85907e6a4cd0fdb5a3e3b07ffd5f45eabefc5612a77002fd3561c9` |
| `patches/psf_compile_c49_2026-10-14/test_c48_on_c49.py` | `d3e589e71b317ee7cea4e4b66f56074f9949c279aaf3aee581ec78ba81fe2c3d` |
| `patches/psf_compile_c48_2026-10-12/psf_compile.py` (comparison) | `0400100f416421dba36c921fa3801f75c37e9e15ea39c22a6639cbc9db976d39` |
| `benchmarks/c49_val.py` (run script) | `063cc1dcc242ed764d03c1c501d086f2308241883161ad2757e6f92dde4daefa` |

**Checks before the run.** 54 tests passed:

- `test_c48_on_c49.py`: c48's 36 tests, run on c49 with both items switched off. c49 then behaves as c48.
- `test_c49.py`: 18 tests of items 84 and 85.
- With the items on, three of c48's 36 tests fail by design. Each asserts a choice by ESP, or equality with c47; they
  are restated by duration in `test_c49.py`.

**Seen before the run (development).**

- On the HW2 snapshot (4 failed qubits), c48 compiles SCALE1's R2 n=144 instances 0-2.
- With 6 more qubits marked failed (seed 1), c48 refuses instances 0 and 2 with SCALE1's exact message, while c49
  compiles them. The 20 MPS shots all gave the known answer.
- Part F's seed 1 is that same configuration. Seeds 2 and 3 are new.
- No duration of c49 on part R's cells has been looked at.

**Run.** `python benchmarks/c49_val.py data/2026-10-14/c49_val/run --part all`, in the work container with
qiskit 2.5.2 and qiskit-aer 0.17.2.

**Predictions** (a cell is one device and one test; durations are as computed by the script, without measurements):

| ID | prediction |
|---|---|
| V1 | Every C49 and C49o output with at most 20 active qubits reproduces the input's distribution within 1e-6, and neither arm refuses any cell. |
| V2 | C49 <= 1.02 x C48 duration on >= 95% of cells, and C49o <= 1.02 x C48o on >= 95%. |
| V3 | C49 <= 1.02 x QK3 duration on >= 90% of cells (C48's count reported). |
| V4 | On every Ising cell (tfim, tfimP): C49 <= 0.5 x QK3 duration. |
| V5 | On HW2's seven reading tests on the marrakesh snapshot: C49 <= 1.02 x QK3 on 7 of 7. |
| V6 | C49 gives the same output on a second call (qaoa_n20_p2, tfim_n16, ba_n20 on every device: 15 of 15). |
| V7 | Compile time: C49 <= 1.5 x C48 + 2 s on >= 90% of cells. |
| V8 | Item 85 (part F): on all 9 configurations, C49 compiles, uses no failed qubit, and gives the known answer on all 20 MPS shots. C48's refusals are reported. |
| V9, V10 | Reported only: duration ratios, ESP C49 / C48, `SELECT_STATS`, `L3C_STATS`, `CUT_STATS`. |

**Rule for the release.** c49 is proposed as the next release if V1, V2, V4, V6 and V8 are confirmed. Those are no
regression, exactness, the Ising gain kept, determinism, and the 144-qubit fix.

- V3 and V5 measure the gain of item 84 and are reported with their verdicts.
- If V5 is refuted, the release text says where c49 is still longer than Qiskit.
- The owner decides the release.

**Not claimed.** A shorter duration has not been shown to give a higher fidelity in general. It was observed after the
fact on HW2 (7 of 7 pairs) and in part S (the same gates reordered). TIME1 is the hardware test for it.
