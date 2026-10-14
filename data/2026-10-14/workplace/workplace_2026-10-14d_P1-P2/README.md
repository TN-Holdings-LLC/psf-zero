# Workplace delivery 2026-10-14d: P1 (items 87-88 on c51) and P2 (AI front end a16)

Requests: 「次の指示2」 P1 and P2, and 「次の指示3」. Offline only: no IBM connection, no commit, no push. Finished
04:57 CEST (`TZ=Europe/Zurich date`). Check the kit with `python check_intake.py`.

## Contents

- **`p1/p1-items-87-88-2026-10-14.md`.** Item 87 (chain brickwork, N1 renumbered) and item 88 (rzz with a Pauli frame)
  on c51.
  - Built by `p1/patch/make_c51_87_88.py`, which makes exact substitutions and appends the new code at the end.
  - 13 new tests pass; c51's 63 tests give 60 passes, and the 3 failures are only `test_version`.
  - Results: 81 cells, all exact, all rzz in range.
    - TFIM: 0.66 (FakeFez) and 0.45 (FakeMarrakesh) of c51's duration.
    - At or below Qiskit level 3 given the brickwork and the same fractional Target in 48/48.
- **`p2/p2-a16-ring-rx-2026-10-14.md`.** a16 = a15 + N2's ring formula as a candidate (item 28) + single-qubit runs as
  one rx pulse and an rzz range guard (item 29).
  - 10 tests pass.
  - On W3's 64 cells: at or below a15 in 64/64; 0.70 of a15 on the plain Target and 0.74 on the fractional one;
    0.53-0.54 of Qiskit level 3.

## Placing in the repository (suggested)

- `p1/patch/psf_compile.py` → `patches/psf_compile_c51_87_88_2026-10-14/`, together with `make_c51_87_88.py`,
  `c51_87_88_code.py` and `test_87_88.py`. Run the test from the repository root.
- `p2/a16/` → `patches/psf_ai_compile_a16_2026-10-14/`. Its test reads psf_compile from the folder above, or from
  `PSF_COMPILE`.
- When merging with c52, home edits only the two VERSION substitutions and the changelog anchor of
  `make_c51_87_88.py`. The rest of the code is appended at the end of the file and does not touch c49's body or
  item 86.
