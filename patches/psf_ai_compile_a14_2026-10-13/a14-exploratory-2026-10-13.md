# Workplace exploration (not pre-registered): candidate psf_ai_compile 2026-10-13.a14 -- tests 9/9; its idle-aware estimate on HW1's committed circuits picks the device's winner in 4 of 6 pairs (rank correlation +0.77, ESP 2 of 6 and -0.09); in noisy simulation of small circuits a14 is level with a12 (2026-10-13)

**Status: exploratory.** The estimate's form was fixed and saved to the Project (design note
`a14-design-2026-10-13.md`, SHA-256 of its first version `3ad2abec...`, 04:12 CEST) before it was computed on any
hardware data. One implementation note (re-scoring placed candidates on their own schedule) was added after the first
smoke run and before the hardware data were used. Nothing below changed the form.

## 1. The candidate

`patches/psf_ai_compile_a14_2026-10-13/psf_ai_compile.py`, built from the adopted a12 by `make_a14.py`. Items 19-22:
device-aware candidates (the release's call with the Target, items 58-64), the idle term (first-order Lindblad,
state-aware, not twirled), the shortest-duration candidates scored, and a failed-element guarantee.

Tests (`test_a14.py`, 9 passed in 21 s, release 2026-10-11.4, core 2026-09-29.1): the versions; without a Target the
same circuits as a12; the idle formula on |0>, |1>, |+>; a hand-checked schedule; exact, on target and free of failed
elements on FakeTorino, FakeKingston and FakeBrussels (ecr); a14's estimate never more than 1% above a12's choice; a
coupler marked failed is avoided. a12's own tests still pass on release 2026-10-11.4 (5/5).

## 2. The estimate on HW1's circuits (`hw1_check.py`)

Target: FakeKingston's structure with the day's dashboard calibration (`data/2026-10-12/hw_analysis/`) for T1, T2,
sx/x/CZ/measure errors and durations. HW1's committed ISA circuits; circuits above 22 active qubits skipped
(qaoa_n20_p2's PSF arm, 25 qubits), tfim_n16 left out (F > 1) and GHZ (identical circuits), as in Addendum 455.

| test | ESP ratio | a14 gate + readout only | a14 with idle | real F ratio |
|---|---|---|---|---|
| qaoa_n12_p2 | 1.053 | 1.056 | 1.470 | 0.990 |
| ba_n14 | 1.286 | 1.325 | 1.047 | 0.780 |
| ba_n16 | 1.339 | 1.420 | 1.061 | 1.014 |
| ba_n18 | 1.048 | 1.079 | 0.812 | 0.744 |
| ba_n20 | 1.224 | 1.177 | 0.874 | 0.808 |
| tfim_n12 | 1.044 | 1.043 | 1.180 | 1.195 |

- Same winner as the device: ESP 2 of 6; a14 without idle 2 of 6; **a14 with idle 4 of 6**.
- Rank correlation with the real ratio: ESP -0.09; without idle -0.26; **with idle +0.77**.
- Over all 18 circuits (sizes differ), every estimate ranks F well: ESP +0.91, gate-only +0.91, with idle +0.94.
- Misses: qaoa_n12_p2 (predicted 1.47, real 0.99) and ba_n14 (1.05 against 0.78; Addendum 455 already noted that
  ba_n14's PSF circuit was shorter and still lost).
- As absolute numbers the estimate is too pessimistic on long circuits (ba_n20 QK3: exp(-4.0) = 0.018 against F =
  0.174). It is first order and meant to rank.
- Six pairs, one day, chosen by HW1's design; this is a consistency check, not a test.

## 3. Noisy simulation of small circuits (`sim_check.py`)

27 circuits (QAOA rings, TFIM chains, model-like circuits with unitary blocks, SWAPs and redundant CX; 4-6 qubits)
on FakeKingston, FakeTorino and FakeFez; Aer noise of the device, circuits scheduled ASAP with delays so idle windows
get relaxation; 8,000 shots; Hellinger fidelity to the ideal distribution.

- a14 against a12: F ratio geometric mean 1.0001 (better 9, worse 12; within shot noise).
- a14 against Qiskit level 3 with the Target: 1.0008 (better 15, worse 12); a12 against level 3: 1.0007.
- The circuits last 3-7 us against T1 of about 100-300 us: idle loss is small here, so a14 has little to choose on.
  The front end's circuits (at most 8 qubits) are where the idle term matters least.

## 4. Reading

1. The idle-aware estimate, with its form fixed from physics, ranked HW1's pairs much better than ESP. This makes it
   a reasonable predictor to pre-register for the hardware replication, not a result.
2. a14 as a front end has not shown a gain on small circuits; its items 19 and 22 (device-aware candidates, no
   failed element) are safety improvements that cost about 0.1-0.3 s per circuit.
3. Not established: anything on hardware for a14; the estimate on another device or day.

## 5. Proposal for the replication (owner's decision)

Pre-register, before any job, a14's estimate (`a14_predict.py` on the compiled circuits and the pickled live Target
of the chosen device) as the predictor: the same winner as the device on more than half of the PSF/QK3 pairs, and a
positive rank correlation. Leave tests whose F does not measure fidelity (states near |0...0>) out of the
predictions in advance.

## 6. Extended (added 2026-10-13, 06:15 CEST)

Section 2's check is extended to all seven pairs, qaoa_n20_p2's 25-qubit PSF arm included, with exact reduced states
from qiskit-aer: 5 of 7, rank correlation +0.75 (`a14-hw1-seven-pairs-2026-10-13.md`). The changelog items were
relabelled 19-22 (design note, section 6); times are now given in CEST.
