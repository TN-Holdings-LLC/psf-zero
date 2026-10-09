# PSF-Zero weakness report, 2026-10-09

**Status: a summary of measured weaknesses, for choosing what to fix. It predicts nothing and scores nothing.** Each
row names its evidence; the Addenda are in Part 10
([`spare-qubit-cliff-combined-383.md`](spare-qubit-cliff-combined-383.md)). TOQB's own weakness report
(`python -m toqb.weakness`, TOQB v12b) will add the six compilers' failures, slow and worse outputs from its runs.

## High

| # | weakness | evidence | status |
|---|---|---|---|
| 1 | **The recommended call's estimates and exactness checks have no bound below 17 qubits.** Their work grows as gates times 2^qubits. | hwb10 (16 qubits) did not finish in an hour; ham_JW-14 took 130-200 s, the ham_enc_gray tests 75-146 s; on the slow tests these functions took 70-100% of the compile (Addenda 412, 415). | Item 54 (candidate c27) bounds them, at a quality cost (Addendum 417). |
| 2 | **On large circuits PSF-Zero's own output loses to Qiskit level 3.** | On all five tests where the unbudgeted call finished, it ended by choosing level 3's circuit; PSF-Zero's circuits had 0.5-19% more two-qubit gates (ham_JW-14 +19%, ham_enc_gray +13% and +16%) (Addendum 417, section 4.2). | Not examined. The largest quality finding of the day. |
| 3 | **The estimates cost ten times as much per amplitude as the checks.** | 46-57 ns against 2-5 ns per amplitude (Addenda 415-416): each two-qubit gate reads its qubits' reduced states from the whole state (item 49). | Not examined. The first place for a faster implementation, Rust included. |

## Medium

| # | weakness | evidence | status |
|---|---|---|---|
| 4 | **The budget is spent in the wrong order.** Re-syntheses whose results then lose are estimated and checked before the choice among candidates. | With a 10 s budget, the work ran out in the first re-synthesis; the choice that would have returned level 3's circuit was never made (Addendum 417, section 4.2). | Proposed: choose among candidates first, or compare cheaply first. Not tested. |
| 5 | **The layout search stops by the clock.** | A 2 s budget, call limits shrunk by the time left, a time-budgeted packing search: the output depends on the machine and its load (Addenda 410-411). | Candidate item 55: budgets in calls. |
| 6 | **Two processes can give different outputs, cause unknown.** | With the layout clock virtual and `PYTHONHASHSEED` fixed: 8 of 185 tests and calls in C25-ID2 (Addendum 412), 2 of 34 tests in C27-B (Addendum 417). | Not examined. A suspect: iteration over objects hashed by identity. |
| 7 | **The same compile varies up to 2.8 times in time within one process.** | REL on ham_enc_gray_dvalues_8-8-8: 104 s, then 37 s (Addendum 414). | Not examined. |

## Low, or addressed

| # | weakness | evidence | status |
|---|---|---|---|
| 8 | networkx's maximum matching took 7-17% of the default call on some tests. | Profile of 2026-10-08 (Addendum 409). | Item 52 (c25): the same outputs (Addenda 410-412). Not adopted. |
| 9 | Building gate matrices was slow. | `_embed_1q` 8.9%, `_ops_of` 7.6% of the recommended call (Addendum 413). | Item 53 (c26): the same values, estimates and checks 1.3-1.5 times as fast (Addendum 414). Not adopted. |
| 10 | A refused estimate still lists the circuit's instructions first. | 3 s on 503,690 instructions; `count_ops` takes 0.008 s (Addendum 417, section 4.1). | Proposed: refuse from `count_ops`. |
| 11 | The default call without a target does not avoid failed elements. | Review of BP-FINAL. | Proposed: a warning. |

## Not yet known

- **Slow compiles above 16 qubits**, where no estimate or check is made: ham_JW-22 66 s, QV_100 30 s in Addendum 415's
  run. Where their time goes has not been measured.
- **`hybrid_cost`'s time** has a part its two counts do not describe (R^2 0.62, Addendum 415).
- **PSF-Zero in TOQB's standard run**: runs 1 and 1b were not scored, and nothing is claimed from them.

## Order proposed

1. Quality: items 2 and 4 together. Comparing with level 3 before re-synthesising could recover both the time and the
   two-qubit gates.
2. Speed: item 3, the estimates.
3. Reproducibility: items 5 and 6, on which every benchmark of PSF-Zero depends.
