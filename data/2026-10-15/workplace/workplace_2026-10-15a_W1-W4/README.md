# Workplace delivery 2026-10-15a: W1, W2, W4 of 「次の指示6（改訂版）」 (base: release 2026-10-14.1 / c57, bundle n)

Finished 04:00 CEST (`TZ=Europe/Zurich date`). Offline only: no IBM connection, no commit, no push. Check the
delivery with `python check_intake.py`. W3 (item 96) needed no work here. C55-VAL was not run.

## Contents

- **`w2/`: RING1 kit (top priority).**
  - **Contents:** `ring1_kit/` (scripts 67-69, `ring1_real.py`, the frozen model, a **draft** pre-registration,
    rehearsals) and the report `w2-ring1-kit-2026-10-15.md`.
  - **What it compares:** a18 against Qiskit level 3 on W3's classifier (n = 4, 6; L = 4, 8; 24 test points each),
    with a ghz_n6 control submitted twice.
  - **Size:** 388 circuits, 800 shots, two repeats. QPU estimate 132 s (conservative) or about 98 s (TIME1's and
    FRAC1's measured rate).
  - **FakeFez:** A18 is 0.48-0.59 times QK3's duration. Every check passed. The noise-model rehearsal is reported
    only.
  - **For home:** the margin is read on one physical qubit, which differs between the arms. RA is therefore the pooled
    ratio alone. Fix the pre-registration, then update its line in `SHA256SUMS`.
- **`w1/`: a18 review.**
  - Outside the scope, a18's output was identical to Qiskit level 3 in 16 of 16 cells (folded on fractional Targets,
    as c57 does).
  - **Main concern:** the hand-off's exactness check builds the statevector from the input as written. On a
    PauliEvolutionGate input it took 960 s for 12 qubits and said "inexact", because it compares against the exact
    exponential, not the product formula. Unrolling first, as item 93 does, takes 0.2 s.
  - Smaller points: two tolerances (1e-9 against 1e-6); the ring path stops at 8 qubits; `seed_transpiler` in kwargs
    raises.
- **`w4/`: item 97 (orientation rule).**
  - `make_c57_97.py` on c57: symmetric two-qubit gates are written smaller index first, in scope only.
  - **Without the rule:** in scope, c57 depended on the orientation in 18 of 100 cells (up to 9.7%). Every one of
    these went through Qiskit level 3 inside c57; CZ chains are not read by items 87 and 91.
  - **With the rule:** one output for every orientation. Longer than some orientation in 16 of 300 (cell,
    orientation) pairs (at most +9.7%), shorter in 15. On average +0.2%. The two-qubit count never changed.
  - **Tests:** 7 new tests pass. c57's tests: 113 pass; 9 fail only on the version string.
  - **DAG order:** Qiskit's DAG is sorted by qubit indices, so orientation is the one written freedom it keeps. The
    written order of commuting gates also matters (up to 2.5%); item 97 does not address that.
