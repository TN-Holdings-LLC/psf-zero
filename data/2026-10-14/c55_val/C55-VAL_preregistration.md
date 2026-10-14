# C55-VAL: pre-registration (fixed 2026-10-14 09:27 CEST, before the run)

**What is tested.** Prototype `2026-10-14.c55` is c54 plus two items (Addendum 486).

- **Item 94 (workplace R1).** On the outermost call, the input's runs of pairwise commuting ZZ terms are regrouped
  into layers of disjoint pairs by a first-fit edge colouring, in their order. c54 compiles both the input and this
  rewrite, and the shorter exact output is returned (item 84's rule, item 93's check). Merged by `make_c55_94.py`; only
  the version strings change.
- **Item 95 (home, after the workplace's R2).** Item 93's check of a measured circuit goes up to 20 active qubits
  (it stopped at RESYNTH_MAX_QUBITS = 16). The tolerance stays 1e-6.

Offline only.

**Files fixed by SHA-256:**

    f2b77f055f88e1199937aa9d6a60db3570e2c6e6e112530ead0e5fdb6dfd4b63  patches/psf_compile_c55_2026-10-14/psf_compile.py
    44e75632361e7993d81c11900e18c91688f858b0da69538ece3a35c2827b4dc9  patches/psf_compile_c55_2026-10-14/make_c55_94.py
    b5cdc7705d05797293263503e032f30bf0645342b87c11c1f75763b2c25dcbe4  patches/psf_compile_c55_2026-10-14/make_c55.py
    21eb723b3cc7f761070f675e67d9055147e569f475b256779c7ccc3aa31b3582  patches/psf_compile_c55_2026-10-14/c55_94_code.py
    99d350747c451fe8177f553a9dfeffa176041b6a1173cd2e5ed3f62af2b49df8  patches/psf_compile_c55_2026-10-14/test_94.py
    c8a61388f6147f342d33978109dac2e7b64ea8b29aea6a1d32b414972978b391  patches/psf_compile_c55_2026-10-14/test_93_on_c55.py
    20cb60a218d625c1eb73ddf0e536a58250c4f2249e8d313ed27ac219942cc28e  benchmarks/c55_val.py

**Checks before the run.** 110 tests passed on c55:

- the workplace's 7 item 94 tests;
- item 93's tests with a new one for item 95 (a 20-qubit circuit is now checked);
- c54's, c53's, c52's, c51's, c49's and c48's suites, with item 94 off where outputs are compared to older versions.

a17's 13 tests pass with c55.

**Seen before the run.**

- The workplace's R1 on c54 + item 94: 22 family-map cells on FakeFez, never longer, 6 shorter (0.83-0.998).
- The workplace's R2 scan: on c54, only tfim_n20_r3's outputs differ from the input's distribution by more than 1e-12
  (1.5e-8 to 3.2e-7), and item 93 did not check them.
- No C55-VAL cell has been run on c55.

**Run.** `python benchmarks/c55_val.py data/2026-10-14/c55_val/run`.

- **Scope.** C54-VAL's 28 tests, on 5 devices.
- **Arms.** QK3 (Qiskit level 3); C54 and C55 without fractional gates; C54F and C55F on the fractional Target.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | C55 and C55F exact (<= 20 active qubits, 1e-6), with no refusal. |
| V2 | C55 <= 1.02 x C54 duration on >= 95% of cells. |
| V3 | C55F <= 1.02 x C54F on >= 95% of cells. |
| V4 | C55 shorter than C54 by >= 1% on >= 10% of cells (without fractional gates). |
| V5 | C55F gives the same output on a second call (the same cells as C54-VAL). |
| V6 | Every C55F output native, with every rzz angle in range. |
| V7 | Compile time: C55F <= 2.5 x C54F + 3 s on >= 90% of cells (item 94 compiles twice). |
| V8 | Item 93 in c55: no fallback and nothing raised; the largest difference passed lies in [1e-9, 1e-6] (item 95 now sees tfim_n20_r3). |

Reported without a threshold: cells where C55 or C55F is longer than c54 by more than 0.1%, family ratios, ZZ94_STATS,
GRAPH_STATS, EXACT93_STATS.

**Rule.** If V1, V2, V3, V5, V6 and V8 are confirmed, c55 becomes the base. V4 decides whether item 94 is called a gain
in writing. TIME1 and FRAC1 tonight are unchanged (c49 and c51).
