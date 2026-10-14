# C52-VAL: pre-registration (fixed 2026-10-14 04:16 CEST, before the run)

**What is tested.** Prototype `2026-10-14.c52` is c51 plus two items.

- **Item 89.** In item 73's reader, when every qubit is measured at the end, a diagonal Clifford left before the
  measurements is dropped. The outcome distribution is unchanged.
- **Item 90.** Items 33/60, 64 and 78 are off by default; the workplace's N5 found them without effect. Item 75's
  re-placement is scored by gate durations instead of errors.

Offline only.

**Files fixed by SHA-256.**

| file | SHA-256 |
|---|---|
| `patches/psf_compile_c52_2026-10-14/psf_compile.py` | `9845dd43cfa523239e3c0391330096f06c1510800c476e7a64a18c41a158253e` |
| `patches/psf_compile_c52_2026-10-14/make_c52.py` | `0d1d2c5fda6690db02d1582d22244102f48babdb55ecb2acb9131dbdf104dd6e` |
| `patches/psf_compile_c52_2026-10-14/test_c52.py` | `4db4a1bb1232852f32486a2d1f8426abcc988a0bdb1498022fe3e4d41df0a571` |
| `benchmarks/c52_val.py` | `4102f048d8aea07d5f723b2933e3d5d83b08d2c1a13077b1c3874481d2547805` |

**Checks before the run.** 73 tests passed:

- c51's, c49's and c48's tests on c52, with items 89 and 90 switched off where they compare outputs;
- `test_c52.py` (10).

**Seen before the run (FakeFez, development).** On W3-type classifiers with odd L, c52 against c51:

| n | L | c51 (us) | c52 (us) |
|---|---|---|---|
| 4 | 3 | 4.38 | 3.28 |
| 6 | 5 | 10.18 | 6.30 |
| 6 | 9 | 17.94 | 11.56 |

All three were exact. The VAL's classifier instances use other seeds.

**Run.** `python benchmarks/c52_val.py data/2026-10-14/c52_val/run`.

- **Scope.** C49-VAL's 18 tests plus six odd-layer classifiers (n = 4, 6; L = 3, 5, 9), on 5 devices.
- **Arms.**
  - QK3: Qiskit level 3.
  - C49 and C52: no fractional gates.
  - C51F and C52F: fractional Target.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | C52 and C52F exact (<= 20 active qubits, 1e-6), with no refusal. |
| V2 | C52 <= 1.02 x C49 duration on >= 95% of cells. |
| V3 | Odd-layer classifiers: C52 / C49 duration geometric mean <= 0.85. |
| V4 | C52F <= 1.02 x C51F on >= 95% of cells. |
| V5 | C52 gives the same output on a second call: qaoa_n20_p2, tfim_n16, ba_n20 and clf_n6_L5 on every device. |
| V6 | Every C52F output native to the fractional Target, with every rzz angle in range. |
| V7 | Compile time: C52 <= 1.5 x C49 + 2 s on >= 90% of cells. |

**Rule.** If V1, V2, V3, V5 and V6 are confirmed, c52 becomes the base of the next prototypes, and the workplace's
items 87-88 are merged onto it. c52 is not run on hardware tonight: TIME1 and FRAC1 stay as fixed, with c49 and c51.
