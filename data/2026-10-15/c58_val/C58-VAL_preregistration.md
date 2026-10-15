# C58-VAL: pre-registration (fixed 2026-10-15 02:24 CEST, before the run)

**Why.** The owner asked for a formula-only prototype, tested for reliability and speed (2026-10-15).

- Measured that morning: inside the scope, the release 2026-10-14.1 took 0.6-37 s per call. Most of that went on
  item 93/95's state-vector checks and on candidates that lose. The formula alone took 27-73 ms.
- a18 took 0.17-1.2 s per classifier, against 28-36 ms for the ring formula alone.

**What is tested.**

- **c58 = release 2026-10-14.1 + item 97.** With the device's Target, an in-scope input is built by item 87's chain
  brickwork or item 91's edge colouring and returned as it is. Nothing else is built or compared, and nothing is
  simulated. If neither formula builds, the input is handed to Qiskit's level 3, as item 96 does.
- **a19 = a18 + item 34.** For a ring classifier, only the two ring schedules are built, and the shorter is returned.
  There are no a15 starts, no polish, and no per-candidate check.

Offline only.

**Files fixed by SHA-256:**

    51ee428b4f72e8f30d74d9ea658761c18e2e84ee14aadb4fc6d678b975fe3f5c  patches/psf_compile_c58_2026-10-15/psf_compile.py
    897c7ba1b3ab804fcdd8a7e9fefd61b20fd7d5af3607ac0eea1fcb86d7ec76d9  patches/psf_compile_c58_2026-10-15/make_c58.py
    4880be812620882c6166a649ce49b623f1c737a98ab06ee50133a76a9b1aa278  patches/psf_compile_c58_2026-10-15/c58_97_code.py
    5ebbb9f12f2921068eae8ef128a5359948a056890666c2373a6050070a414f0c  patches/psf_ai_compile_a19_2026-10-15/psf_ai_compile.py
    84ed65344ebd8862ccc5a00e5e24139d8dfa886eb9cf4169d1b4bee95e98b5f6  patches/psf_ai_compile_a19_2026-10-15/make_a19.py
    ff69e367a714c7e8ba55ab1fdfe5d4c05b34e010650346f9122bd6fd126e1707  benchmarks/c58_val.py
    ca359966031d67fd99c423b6200ff20343ead995a9a373cb6fc6b8f40789e829  data/2026-10-14/c57_val/run/c57_val_rows.jsonl

**Seen before the run.**

- The timings above, on seven in-scope tests on FakeFez: the release, the formula alone, and Qiskit.
- One smoke call each of c58 and a19, on a 9-qubit chain and a 5-qubit ring on FakeKingston (not in this run's
  cells).

**Run.** `python benchmarks/c58_val.py data/2026-10-15/c58_val/run`.

- **Part A:** C57-VAL's 28 tests on its 5 devices, T and F (280 cells), compared with C57-VAL's rows for the release.
- **Part B:** W3's 64 classifier cells, T and F (128 cells), compared with a18's rows.

**Predictions.**

| ID | prediction |
|---|---|
| P1 | Every c58 and a19 output exact (<= 1e-6, checked by the harness where <= 20 qubits are active), native, every rzz in range, on no failed element. |
| P2 | Out of scope: c58 identical to the release on every cell. |
| P3 | In scope: c58's compile time <= 0.1 x the release's on >= 90% of cells. |
| P4 | In scope: duration with fractional gates (F) <= 1.01 x the release's on >= 90%; without them (T) <= 1.5 x on every cell. |
| P5 | a19: duration equal to a18's on >= 62 of 64 cells per Target, and compile time <= 0.2 x a18's on >= 90%. |
| P6 | The same output on a second call (5 tests x 5 devices x 2). |

**Rule.** Exploratory prototype; nothing is released on this result alone. If P1, P2 and P6 hold, formula-only is safe
to build on. P3-P5 say what it gains and costs.
