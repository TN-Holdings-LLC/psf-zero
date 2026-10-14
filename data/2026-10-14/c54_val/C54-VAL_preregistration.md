# C54-VAL: pre-registration (fixed 2026-10-14 07:54 CEST, before the run)

**What is tested.** Prototype `2026-10-14.c54` is c53 plus two items (Addendum 485).

- **Item 91 (workplace Q1).** An interaction graph that embeds in the device (first VF2 match, no score) is built by
  edge colouring: one layer per colour (at most 3 on heavy-hex) and a single-qubit layer, with item 88's rzz / rx
  Pauli frame when the Target has them. Merged by `make_c54_91.py`; only the version strings change.
- **Item 93 (home, after the workplace's Q3).** Every output with at most RESYNTH_MAX_QUBITS active qubits is checked
  exact at 1e-6 (largest outcome-probability difference; state infidelity when unmeasured). If it is not exact, the
  usual path and then Qiskit's level-1 preset are tried; if neither is exact, TranspilerError is raised.

Offline only.

**Files fixed by SHA-256:**

    5295733e388b91a6ae1c28c57210608c023875512b57b431c55da9eb4f10e63b  patches/psf_compile_c54_2026-10-14/psf_compile.py
    127e4cf2271afb157f9545862a10b78da9fc5933110033e530484216ed78f0b1  patches/psf_compile_c54_2026-10-14/make_c54_91.py
    7e7790e4729e06f2671c7ec8cafbf1559f8120c3b57f107fc8ab678a29c406bd  patches/psf_compile_c54_2026-10-14/make_c54.py
    4e7611acebd1ff3b110ba11683690cca6f77fd04a4ed56e48d0c4e02bdd8c114  patches/psf_compile_c54_2026-10-14/c54_91_code.py
    850d86a9e681e479a992643147aff25ec205fb9e5d8f8d4c14798c3714c9533a  patches/psf_compile_c54_2026-10-14/c54_93_code.py
    bb05db8cf6c908292c11c70db491e930e554ea9d076bf21cf4c14e9f92a375cf  patches/psf_compile_c54_2026-10-14/test_91.py
    646d2026d4b00ee1ee8ee46829a37e11156839ff2dd0a2ad12f040b430209896  patches/psf_compile_c54_2026-10-14/test_c54.py
    933c54222c2231f371e05d2c8440f4bea206ff854f3c2dd1bf855b7c1ebbf612  benchmarks/c54_val.py

**Checks before the run.** 102 tests passed on c54: the workplace's item 91 tests, item 93's 6 new tests (a snapped
block from the workplace's fixture is detected and replaced by an exact output; the switch off restores c53; an
unmeasured circuit is checked; a large circuit is not checked), and c53's, c52's, c51's, c49's and c48's tests. a17's
13 tests pass with c54.

**Seen before the run.**

- The workplace's Q1 evaluation on c53 + item 91 (heavy-hex Ising at 0.62-0.74 x c53's duration).
- A dry check of three of the new cells on FakeFez, C53F and C54F only: ring12_tfim 1.104 -> 0.996 us, hhI_n16 1.620
  -> 1.392 us, hhtree_heis_n16 5.508 -> 5.508 us (unchanged), all exact. V4's threshold is the one planned before
  the dry check and is not adjusted.

**Run.** `python benchmarks/c54_val.py data/2026-10-14/c54_val/run`.

- **Scope.** C53-VAL's 24 tests plus 4 embeddable ones (hhI_n16, hhI_n20: Ising on a 16 / 20-qubit heavy-hex patch
  of FakeFez's graph; ring12_tfim; hhtree_heis_n16: Heisenberg on a 16-qubit heavy-hex spanning tree), on 5 devices.
- **Arms.** QK3 (Qiskit level 3); C53 and C54 without fractional gates; C53F and C54F on the fractional Target.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | C54 and C54F exact (<= 20 active qubits, 1e-6), with no refusal. |
| V2 | C54 <= 1.02 x C53 duration on >= 95% of cells. |
| V3 | C54F <= 1.02 x C53F on >= 95% of cells. |
| V4 | Embeddable tests on the fractional Target: C54F / C53F geometric mean <= 0.85. |
| V5 | C54F gives the same output on a second call (the same cells as C53-VAL). |
| V6 | Every C54F output native, with every rzz angle in range. |
| V7 | Compile time: C54F <= 1.5 x C53F + 2 s on >= 90% of cells. |

Reported without a threshold: family ratios, GRAPH_STATS, EXACT93_STATS (how often item 93 fell back or raised, and
the largest difference passed).

**Rule.** If V1, V2, V3, V5 and V6 are confirmed, c54 becomes the base. V4 decides whether item 91 is called a gain
in writing. TIME1 and FRAC1 tonight are unchanged (c49 and c51).
