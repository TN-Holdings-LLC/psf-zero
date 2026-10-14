# C50-VAL: pre-registration (fixed 2026-10-14 03:39 CEST, before the run)

**What is tested.** Prototype `2026-10-14.c50` is c49 plus item 86. When the device's Target offers IBM Heron's
native fractional gates (rzz, a ZZ rotation in one CZ-length pulse; rx in one pulse), the finished circuit is
re-synthesised with them on its own qubits and layout. The result is kept when it is shorter and exact.

- The re-synthesis runs on an error-free copy of the Target, so no reported error chooses.
- This is offline only.
- Whether rzz runs as well as CZ on the device is a separate hardware question (FRAC1).

**Basis.**

- ibm_marrakesh's properties of 2026-10-13 list rzz at 68 ns (the CZ's length) and rx at 36 ns (an sx's).
- IBM's fractional-gates guide: rzz needs 0 < theta <= pi/2 (FoldRzzAngle folds fixed angles), rx takes any angle, and
  Pauli twirling is not supported.
- The fake devices have no fractional gates. Their copies here add rzz on every cz coupler with the CZ's duration,
  and rx on every qubit with the sx's duration.

**Files fixed by SHA-256.**

| file | SHA-256 |
|---|---|
| `patches/psf_compile_c50_2026-10-14/psf_compile.py` | `c57eba801373cd1993c21afe24b9375b5754d853073784875efc0958bf05e4a1` |
| `patches/psf_compile_c50_2026-10-14/make_c50.py` | `982db638650e48b026b12d723a99f35ebaf1a50c4f52ab73ab393a9cef15f1cb` |
| `patches/psf_compile_c50_2026-10-14/c50_code.py` | `89ea01a7653610f2d9dc990b239a6b8617efe910922cd1188c8bcee0f7439645` |
| `patches/psf_compile_c50_2026-10-14/test_c50.py` | `dba1cbb33b27ae5674fb2dc3f3a2522a80d8e7fe9bfe1f53404cbf8531d19f6e` |
| `patches/psf_compile_c50_2026-10-14/test_c49_on_c50.py` | `c01b77dcccdcba8f16e546f9ad643d2b4ec7825e1eb3950645e043080f4c8bfe` |
| `patches/psf_compile_c50_2026-10-14/test_c48_on_c49_for_c50.py` | `061838bd748410f96db3d5cc7193915bd89d9277c76c30f2f65b0409f04b4d6c` |
| `patches/psf_compile_c49_2026-10-14/psf_compile.py` (comparison) | `5544d7c9f10d615c041eaf273af311573151a2548690b2192d866509bdaaa8d0` |
| `benchmarks/c50_val.py` (run script) | `3766f9daae8a3aa1ea6d6a8d76a930423d16d9ffeae28ceff29062ceafd28aa8` |

**Checks before the run.** 62 tests passed:

- 54 of c49's and c48's tests, run on c50. Their Targets have no rzz, so c50 must behave as c49.
- `test_c50.py`, 8 tests:
  - without rzz the output is c49's;
  - Ising uses rzz, is exact and shorter, with every angle in range;
  - never longer and exact on small circuits;
  - a measured circuit keeps its distribution;
  - rescaled errors do not change the re-synthesis;
  - deterministic, and `FRACTIONAL = False` gives c49's output;
  - failed qubits are still cut.

**Seen before the run (development, FakeFez with rzz/rx, c49's output re-synthesised by the same passes).**

| circuit | c49 (us) | c49 + fractional (us) | ratio |
|---|---|---|---|
| tfim_n12 | 1.49 | 0.82 | 0.55 |
| tfimP_n16 | 1.51 | 0.82 | 0.54 |
| tfim_n40_r2 | 1.06 | 0.91 | 0.86 |
| qaoa_n20_p2 | 6.31 | 5.27 | 0.84 |
| ba_n20 | 24.4 | 22.1 | 0.91 |
| heis_n12_r2 | 16.7 | 17.6 | 1.05 |

The heis_n12_r2 result would not be kept under item 84.

**Run.** `python benchmarks/c50_val.py data/2026-10-14/c50_val/run`, with qiskit 2.5.2 and qiskit-ibm-runtime
0.49.0.

- **Scope.** C49-VAL's 18 tests on 5 devices: FakeFez, FakeMarrakesh, FakeKingston, FakeTorino, and the marrakesh
  snapshot.
- **Arms.**
  - QK3: Qiskit level 3, no fractional gates.
  - QK3F: Qiskit level 3 on the fractional Target.
  - C49: today's call, with no fractional gates.
  - C50: on the fractional Target.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | Every C50 output with at most 20 active qubits reproduces the input's distribution within 1e-6, and C50 refuses no cell. |
| V2 | C50 <= 1.02 x C49 duration on >= 95% of cells. |
| V3 | On the Ising cells, C50 / C49 duration has a geometric mean <= 0.70, and every Ising cell is <= 0.95. |
| V4 | C50 <= 1.02 x QK3F on >= 90% of cells. |
| V5 | C50 gives the same output on a second call: qaoa_n20_p2, tfim_n16 and ba_n20 on every device. |
| V6 | Every C50 output is native to the fractional Target, and every rzz angle is in (0, pi/2]. |
| V7 | Compile time: C50 <= 1.5 x C49 + 2 s on >= 90% of cells. |
| V8, V9 | Reported only: ratios by family, `FRAC_STATS`. |

**Rule.** If V1, V2, V3, V5 and V6 are confirmed, item 86 goes to the hardware test FRAC1. That test compares the same
inputs compiled with and without fractional gates, on the device. Item 86 is not proposed for release before FRAC1
and TIME1.
