# Q3 (2026-10-14): a15's inexact output came from Qiskit's two-qubit synthesis snapping a block (1e-9 gate infidelity), not from PSF-Zero's tolerances; fixed in the AI front end by checking every candidate (a17 item 30)

**Status: cause found, fix with tests and rerun; offline only.** Workplace, 2026-10-14, finished 07:36 CEST
(`TZ=Europe/Zurich date`). Request: 「次の指示4」 Q3.

## 1. Locating it (scripts `q3_probe*.py`)

**The cell.** a15 on FakeFez with rzz and rx, classifier n = 6, L = 12, W3's angles. a15 returned the candidate
"commuted" at total variation distance (TVD) 1.39e-5 from the input.

**Step by step:**

1. **The commuted start itself** is exact (TVD 8e-16).
2. **psf_compile c51 on it** gives TVD 1.39e-5 with `FRACTIONAL = "auto"`. Item 86's re-synthesis was "longer", so it
   was not used.
3. **With `FRACTIONAL = False`** the TVD is the same 1.39e-5, and only the fold acted. So it is **not** item 86's
   re-synthesis.
4. **c49's part, before the fold,** is already at 1.39e-5. So it is **not** the fold, and not a14's polish either (the
   chosen candidate was not polished).
5. **Wrapping every `transpile` call inside c49:** the STEINER6 path (items 68, 73, 82-83, used here) ends in
   `transpile(phys, target, initial_layout=range(N), routing_method="none", optimization_level=3)`. Its input is exact;
   its output has TVD 1.39e-5.
6. **The same call on other Targets and levels:**

   | Target / level | TVD |
   |---|---|
   | plain FakeFez Target (no rzz) | 1.39e-5 |
   | error-free copy of the Target | 1.39e-5 |
   | `approximation_degree=1.0` explicit | 1.39e-5 |
   | level 2 | 1.39e-5 |
   | level 1 | **2e-15** |

7. **A per-pass callback at level 3:** the jump is at **`TwoQubitPeepholeOptimization`** (Qiskit 2.5.2, pass 24).
8. **Block by block:** of 121 two-qubit blocks, one comes back wrong. It has gate infidelity **8.46e-10** and Weyl
   coordinates (0.1489, 0.1489, 0) — a = b and c = 0, a degenerate point. `TwoQubitBasisDecomposer(CZ)` gives the same
   8.46e-10 with `approximate=False` and with `basis_fidelity=1.0`, and so do `ConsolidateBlocks + UnitarySynthesis`.

**Cause.** Qiskit's `TwoQubitWeylDecomposition` snaps a unitary to a specialised form when the specialisation costs at
most 1e-9 in average gate fidelity (its default `fidelity = 1 − 1e-9`). This is independent of `approximation_degree`
and of the Target's errors. A gate infidelity of 8.5e-10 is an amplitude error of about 3e-5, and here a TVD of 1.4e-5
in the output distribution.

## 2. Why no check caught it (home's two guesses)

- **PSF-Zero's item 39 check** (`_same_action`, EXACT_TOL = 1e-6 in state infidelity) is too loose for this. An
  infidelity of 1e-6 allows amplitude errors up to about 1e-3, so a 1e-9 snap passes easily. On top of that, the
  STEINER6 path's final transpile is not checked at all.
- **a14's polish** verifies at tol = 1e-5, also too loose. It was not involved in this cell.
- **So both guesses point at the right weakness** — tolerances too loose to see it — **but neither is the source.** The
  source is in Qiskit, in every level-2/3 compile with two-qubit re-synthesis.
- **C52-VAL already shows the same order of error.** tfim_n20_r3 has a largest per-outcome probability difference of
  1.5e-8 (C52) and 3.2e-7 (C52F) on all five devices. That is likely the same mechanism (not traced here).

## 3. Fix in the AI front end: a17 item 30 (`make_a17.py`, exact substitutions on a16)

- Every candidate is checked against the input before item 25 compares them:
  - **measured input:** the outcome distribution over the input's clbits, computed exactly (statevector of the active
    qubits, at most 20), within **1e-9 in TVD**;
  - **unmeasured input:** item 39's `_implements` at a state infidelity of **1e-12** instead of 1e-6.
- An inexact candidate is dropped. One that cannot be checked stands, and is counted.
- This is a correctness gate, not a choice: it never ranks candidates. Item 25 still chooses by duration.

**On the a15 cell:**

| configuration | chosen | TVD | duration |
|---|---|---|---|
| without item 30 | "commuted" | 1.4e-5 | 10.90 µs |
| with item 30 (both inexact candidates dropped) | "input" | 6.9e-14 | 11.95 µs |

## 4. a17 on W3's 64 cells (c53; T and F)

**Durations.**

- a17 = a16 on 128/128 cells; durations are identical.
- 0.70× a15 (T) and 0.74× (F).
- 0.54× QK3 and 0.53× QK3F; at or below Qiskit in 64/64 on both.

**Exactness.**

- Largest TVD of a17's outputs is 7e-14.
- Every rzz is in range.

**What item 30 saw** (`q3_count.log`): 896 candidates checked, **2 dropped**. Both are the same cell's "commuted" and
"commuted+polish" (FakeFez F, n = 6, L = 12, TVD 1.4e-5). Neither was chosen there, because item 28's ring is shorter
and exact. So on c53, a16's outputs were already exact. a17 makes that a guarantee rather than luck.

**a17 also carries item 31** (a second ring schedule, see `q2-q4-rings`). It changes nothing on W3's n = 4 and 6.

**Tests** (`test_a17.py`, 13 tests, all pass): home's a16 tests on a17, plus:

- an inexact candidate (an extra rx(2e-3)) is dropped, both measured and unmeasured, while the exact one stays;
- the new ring schedule covers every ring with n = 4-60;
- at n = 7, "ring2" is chosen and is exact.

## 5. For home (compiler side, not done here)

- **Check the outputs of exact paths at a tight tolerance.** Paths that end in a Qiskit level-2/3 transpile (STEINER6's
  final transpile, item 80's level-3 candidate, the usual path) can return circuits about 1e-5 away in distribution.
  A proposal for a later item:
  - check the final circuit at ≤ RESYNTH_MAX_QUBITS against the input, by distribution or by `_implements` at 1e-12;
  - on failure, redo that path's final translation at optimization level 1, which was exact here (13.56 µs against
    10.90, so only as a fallback).
- **EXACT_TOL = 1e-6 (state infidelity)** should be tightened to about 1e-12 wherever it gates "exact". Item 86's
  re-synthesis is kept only if `_same_action` passes, so it can accept a snapped block too.

## 6. Files

- `a17/`: `make_a17.py`, the built `psf_ai_compile.py`, `test_a17.py`, `test_a17_result.txt`.
- `probe/`: `q3_probe.py` to `q3_probe7.py`, `q3_gate_check.py`, `q3_count.py`, `q3_count.log`, `bad_block.qpy` (the
  snapped two-qubit block).
- `eval/`: `p2_a16_{T,F}.jsonl` and `p2_a17_{T,F}.jsonl` (both on c53), `summary.txt`.
