# Workplace delivery 2026-10-14e: Q1-Q4 of 「次の指示4」 (base: home's c53 and a16 from bundle e)

Finished 07:36 CEST (`TZ=Europe/Zurich date`). Offline only: no IBM connection, no commit, no push. Check the kit with `python check_intake.py`. Item numbers
continue from 91 as asked; the AI front end items continue from 30.

## Contents

- **`q1/`: item 91, embeddable interaction graphs by edge colouring (recommended patch `make_c53_91.py` on c53).**
  - First VF2 embedding, no score; at most 3 colour layers; uses items 87-88.
  - Results: 84 cells on FakeFez and FakeMarrakesh, never longer than c53.
    - Heavy-hex Ising: 0.62-0.74 of c53's duration (geometric mean), and 0.23 on the whole 154-qubit device with rzz.
  - Exactness: exact for n ≤ 24; stabilizer twins up to 154 qubits.
  - C52-VAL's heis2d does not embed (degree 4).
  - Tests: 10 new pass. c53's 86: 80 pass, 5 `test_version`, and 1 ratio test whose baseline item 91 improves
    (explained in the doc).
- **`q2q4/`: Q2 stopped after the check.** Item 92 (rings by N2's formula in the compiler) is exact but longer than c53
  for n ≥ 6: c53 builds commuting ring terms with parity gadgets, without SWAPs. It is not recommended; the code is
  kept as a record.
  - **Q4:** a new ring schedule for any n, with CZ-depth 9 and about 3n CZ per layer (N2's: 55 CZ at n = 7, 203 at
    n = 11), used by a17.
- **`q3/`: the inexactness found in a15** (total variation 1.4e-5) comes from Qiskit's two-qubit synthesis.
  - Mechanism: `TwoQubitWeylDecomposition` snaps a degenerate block when the snap costs ≤ 1e-9 in gate fidelity, even
    at approximation_degree = 1.0. Here it happened in STEINER6's final level-3 transpile.
  - PSF-Zero's EXACT_TOL = 1e-6 is too loose to see it.
  - **Fix: a17** (`make_a17.py` on a16).
    - Item 30: every candidate is checked exactly (distribution within 1e-9, or a unitary at 1e-12).
    - Item 31: the Q4 ring schedule as a second ring candidate.
  - On W3's 64 cells (c53), a17 = a16, all exact. 2 inexact candidates were dropped out of 896.
  - Tests: 13 pass.
  - A compiler-side proposal for home is in section 5 of the doc.

## Placing in the repository (suggested)

- `q1/patch/psf_compile.py` → `patches/psf_compile_c53_91_2026-10-14/`, with `make_c53_91.py`, `c53_91_code.py` and
  `test_91.py`.
- `q3/a17/` → `patches/psf_ai_compile_a17_2026-10-14/`. Its test reads psf_compile from `PSF_COMPILE` (c53 by
  default, as home's a16 test).
- Evaluation scripts run from the repository root. `PSF_NEW` points to the built c53+91 file when it is not at the
  suggested place.
