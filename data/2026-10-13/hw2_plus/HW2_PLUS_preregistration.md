# HW2-PLUS -- additional predictors for HW2, fixed before HW2 runs

Fixed: 2026-10-13 07:46:44 CEST (Europe/Zurich). Committed (Addendum 466) before HW2's job exists.

**Why.** Two workplace documents of 2026-10-13 (Addendum 465) raised points about the predictors already fixed for
HW2:
- the review of ns_sim.py: idle relaxation uses an echo T2; ALAP is an assumption; preparation error is counted twice
  on measured qubits;
- the BA diagnosis: low-T2 qubits and the state-blind exposure.

The ASAP variant of NS1 (Addendum 464) ranked HW1 better than ALAP. HW2's fixed predictors (R3, NS2) stay as they
are. These four are added, each computed after the run from HW2's circuits and the snapshot of its calibration
(`92s_ns_snapshot.sh`), with code fixed now:

| ID | predictor | code (sha256) |
|---|---|---|
| P1 | ns_sim2, as soon as possible, no preparation error on measured qubits, reported T2 while waiting | `ns_sim2.py` 539678f8..., `hw2_predictors.py` f7bb671c... |
| P2 | as P1, with T2 / 2 while waiting (unechoed dephasing) | same |
| P3 | a14's estimate (gate + readout + state-aware idle), exact reduced states (workplace `est_lc.py` bd5bfe62..., a14 `psf_ai_compile.py` 55a502c3...), on a Target built from the snapshot (`snapshot_target.py` 330ca074...) | same |
| P4 | exposure: sum over used qubits of (end of circuit - first gate) / T2, as soon as possible | same |

`ns_sim2.py` with its defaults gives `ns_sim.py`'s numbers (`test_ns_sim2.py`, sha256 adcffeb5..., 3 passed with
ns_sim.py's 6).

**Tests scored:** HW2's seven reading pairs (PSF / QK3), as R2/R3 and NS2. Part S pairs are reported.

**Predictions:**

| ID | prediction |
|---|---|
| Q1 | at least one of P1-P4 is useful: the device's winner on >= 5 of 7 and rank correlation >= 0.5 |
| Q2 | H-T2*: P2's rank correlation >= P1's |
| Q3 | P4 (state-blind) ranks at least as well as P3 (state-aware): rank correlation P4 >= P3 |
| Q4 | reported: each predictor's winner count and rank correlation; part S |

**Reading, fixed now.** A predictor is carried into PSF-Zero's choices (items 69, 80) only if it is useful on HW2.
Its HW1 numbers are reported beside it, never instead. If none is useful, no time-aware predictor is used until HW3
says what the time effect is. Several predictors are tested at once: any one being useful by chance is possible, and
the record will say how many were tried.
