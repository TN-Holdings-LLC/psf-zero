# HW2-PLUS, amendment 2: two corrected simulation predictors P1c and P2c, added before HW2 runs (2026-10-13)

**Status:** fixed before HW2's job exists. The time and the hashes are at the end. The first prestatement
([`HW2_PLUS_preregistration.md`](HW2_PLUS_preregistration.md), Addendum 466) and its code are unchanged and stay in force.

## 1. Why

The workplace's review of the HW3 kit (2026-10-13, section 4) found that `ns_sim.py` and `ns_sim2.py` never charge
the waiting from a qubit's last gate to the measurement. This was checked here on HW1's circuits:

- As soon as possible, which is how P1 and P2 schedule, the uncharged window is 7-36% of the measured qubits' waiting
  (52% in GHZ).
- It differs between the arms of a test (tfim_n12: PSF 31%, QK3 13%), so it can move the predicted ratios.

P1 and P2 were fixed with their code before this was found, and they are scored as fixed. P1c and P2c are the same
predictors with that window charged. This amendment adds them now, after the review and before HW2's results; the
record says so.

## 2. Code

- [`benchmarks/ns_tail.py`](../../../benchmarks/ns_tail.py) runs `ns_sim2.py` unchanged and adds one channel per measured
  qubit: thermal relaxation for (total duration - end of its last gate), with T2 scaled as for the other waiting
  windows (capped at 2 T1).
  - Qubits that are not measured are not charged; a channel there does not change the measured state.
  - Measured qubits with no gate stay in |0> and are not charged.
- [`benchmarks/test_ns_tail.py`](../../../benchmarks/test_ns_tail.py) checks four things:
  - without a window, the result is identical to `ns_sim2.py`;
  - the window's length;
  - the decay of a |1> over the window, e^(-w / T1), against an exact density matrix;
  - that `ns_sim2` is restored after a call.
- [`benchmarks/hw2_predictors_c.py`](../../../benchmarks/hw2_predictors_c.py) takes the circuits, ideal
  distributions, stopping rule and scoring from `hw2_predictors.py`, imported unchanged.

| ID | predictor |
|---|---|
| P1c | as P1 (as soon as possible, no preparation error on measured qubits, reported T2 while waiting), the window before the measurement charged |
| P2c | as P2 (T2 / 2 while waiting, including that window), the window charged |

Run after HW2, after `hw2_predictors.py`:

    python benchmarks/hw2_predictors_c.py <out dir> data/2026-10-13/hw2 --with <P1-P4 out dir>/hw2_predictors.jsonl

## 3. Predictions (HW2's seven reading pairs, the same scoring as Q1-Q3)

| ID | prediction |
|---|---|
| Q4 | P1c ranks at least as well as P1 (rank correlation of predicted and measured PSF / QK3 ratios) |
| Q2c | P2c ranks at least as well as P1c (the unechoed-dephasing hypothesis, on the corrected model) |
| reported | P1c and P2c: same winner as the device on how many of 7, rank correlation, "useful" (>= 5 of 7 and >= 0.5) as defined for Q1; part S ratios |

**Reading, fixed now.**

- Q1 (at least one of P1-P4 useful) keeps its four predictors; P1c and P2c are not added to it.
- With P1c and P2c, six simulation or exposure predictors are scored on HW2. The record will say so when one of them
  succeeds.
- If Q4 is refuted, charging this window, though physically real, did not help on HW2: the time sensitivity is then
  not explained by relaxation alone.
- NS2 (Addendum 462) has the same gap, under ALAP (0-11% of waiting on HW1, except tfim_n12/16 PSF 33% and GHZ). It
  is scored as fixed, and this is stated with its result.

## 4. Fixed

2026-10-13 08:34:13 CEST (Europe/Zurich), before HW2's job exists. SHA-256:

| file | SHA-256 |
|---|---|
| `benchmarks/ns_tail.py` | `aaf16d8e8a2ccf519f697bfdcde11004d988387622779025ca334f2aaa35eb50` |
| `benchmarks/test_ns_tail.py` | `25483fce568ee725ea8c30b1bbc8ac972ba9b2fc1e2c2ef8e2c41907bea640e7` |
| `benchmarks/hw2_predictors_c.py` | `098b14f49a4b5b606540a358f3463db88a048d6d3bb09a5bb2a95b17440b13b5` |
| `benchmarks/ns_sim2.py` (unchanged, Addendum 466) | `539678f82fa374a1251368c636ee217b5c5ce5f14a2b39de466d5f2dd6af5cbf` |
| `benchmarks/hw2_predictors.py` (unchanged, Addendum 466) | `f7bb671cbd614bcf76e9d640229d736dc3e659af2a59254cac451764c0d0d485` |

Tests: `test_ns_tail.py`, `test_ns_sim2.py`, `test_ns_sim.py`, 14 passed.
