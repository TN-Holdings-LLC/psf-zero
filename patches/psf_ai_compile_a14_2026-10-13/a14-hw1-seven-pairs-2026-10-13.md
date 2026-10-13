# Workplace exploration (not pre-registered), requests 1 and 4 of 2026-10-13: a14's idle-aware estimate on all seven HW1 pairs, qaoa_n20_p2 included, picks the device's winner in 5 of 7 (rank correlation +0.75; ESP 3 of 7, -0.18). The light-cone rule R6/16 gives the same 5 of 7 (+0.64). On the BA circuits, the PSF arms that lost used more low-T2 qubit time (2026-10-13)

**Status: exploratory, after HW1's scoring (Addendum 455).** Times in CEST. The idle term's form was fixed at
04:12 (design note); the approximation rule R6/16 was fixed and saved at 05:40 (`a14-lightcone-rule-2026-10-13.md`,
SHA-256 `ea80a2f1...`) before any number was computed with it. Item numbers follow the relabelling (19-22).

## 1. Method (request 1)

- **Target:** FakeKingston's structure with the day's dashboard calibration (T1, T2, sx/x/CZ/measure errors and
  durations), as in the first a14 check.
- **Exact estimate for every circuit, the 25-qubit ones included:** qiskit-aer, `save_density_matrix` after each
  instruction on the compact circuit of the active qubits (statevector, truncation off), the reduced states then fed
  to a14's own formulas (`est_lc.py`). Checked against a14's state-vector code on the 20 circuits of at most 22
  active qubits: largest difference 1.8e-14. qaoa_n20_p2's PSF arm (25 qubits) took 527 s.
- **Rule R6/16** for comparison: exact light-cone states while each qubit has had at most 6 two-qubit gates and the
  cone touches at most 16 qubits; maximally mixed otherwise.
- **Pairs:** PSF against QK3 on the seven tests of Addendum 455 (tfim_n16 left out because F > 1 there; GHZ left
  out because the two circuits are identical).

## 2. The seven pairs

| test | ESP ratio | a14 gate + readout | **a14 with idle (exact)** | a14 with idle (rule) | real F ratio |
|---|---|---|---|---|---|
| qaoa_n12_p2 | 1.053 | 1.056 | 1.470 | 1.476 | 0.990 |
| qaoa_n20_p2 | 1.053 | 1.074 | **1.356** | 1.293 | **1.202** |
| ba_n14 | 1.286 | 1.325 | 1.047 | 1.210 | 0.780 |
| ba_n16 | 1.339 | 1.420 | 1.061 | 1.495 | 1.014 |
| ba_n18 | 1.048 | 1.079 | 0.812 | 0.837 | 0.744 |
| ba_n20 | 1.224 | 1.177 | 0.874 | 0.891 | 0.808 |
| tfim_n12 | 1.044 | 1.043 | 1.180 | 1.329 | 1.195 |

| estimate | same winner as the device | rank correlation with the real ratio |
|---|---|---|
| ESP | 3 of 7 | -0.18 |
| a14 without idle | 3 of 7 | -0.39 |
| **a14 with idle, exact** | **5 of 7** | **+0.75** |
| a14 with idle, rule R6/16 | 5 of 7 | +0.64 |

- qaoa_n20_p2, one of PSF's two clear wins on the device, is predicted as a PSF win (1.36; real 1.20).
- The two misses are the same as before: qaoa_n12_p2 (predicted 1.47, real 0.99, a tie on the device) and ba_n14
  (predicted 1.05, real 0.78).
- Seven pairs from one device and one day, examined after the fact: a consistency check, not a test.

## 3. How good is the rule (request 1, validation)

Per circuit, the rule's total against the exact total (all 22 HW1 circuits, three of them at 25 qubits):

- It always overestimates (it replaces partly coherent states by mixed ones): +0.0% (GHZ) to +33.7% (tfim_n16 QK3),
  median about +6%.
- Largest errors on the TFIM circuits (+7% to +34%), as stated in advance for states that stay near |0...0>, and on
  long QK3 arms (ba_n16 +15.8%, ba_n20 +13.9%).
- The error is not the same for both arms of a pair, so ratios move: ba_n16 1.495 (rule) against 1.061 (exact),
  tfim_n12 1.329 against 1.180. The winner happened not to change in any of the seven pairs.
- Reading: R6/16 is usable for ranking when no exact computation is possible, but exact reduced states (Aer, up to
  about 25-28 active qubits on this machine) should be preferred; the rule's bias depends on how product-like the
  circuit stays.

## 4. BA losses: which qubits and couplers each arm used (request 4)

From HW1's committed circuits and the day's CSV (`ba_diag.py`, output `ba_diag.txt`). Weights: number of non-virtual
gates per qubit, number of CZ per coupler. Device references: T2 20th percentile 60 us, CZ error 80th percentile
0.0032.

| test | arm | real F | CZ count | use-weighted T1 / T2 (us) | lowest T2 used (us) | qubits below T2 p20 | use-weighted CZ error | max CZ on one coupler | exposure sum(window/T2) |
|---|---|---|---|---|---|---|---|---|---|
| ba_n14 | QK3 | 0.263 | 391 | 268 / 168 | 30 | 1 of 15 | 0.0016 | 50 | 2.16 |
| ba_n14 | PSF | 0.205 | 303 | 263 / 169 | 20 | 2 of 14 | 0.0015 | 49 | **2.64** |
| ba_n16 | QK3 | 0.280 | 477 | 230 / 135 | 20 | 3 of 16 | 0.0017 | 55 | 4.15 |
| ba_n16 | PSF | 0.284 | 378 | 262 / 173 | 20 | 2 of 16 | 0.0015 | 52 | 3.90 |
| ba_n16 | C47 | 0.297 | 380 | 274 / 169 | 71 | 0 of 16 | 0.0014 | 58 | **1.93** |
| ba_n18 | QK3 | 0.105 | 584 | 265 / 165 | 30 | 1 of 18 | 0.0016 | 61 | 3.27 |
| ba_n18 | PSF | 0.078 | 531 | 268 / 170 | 30 | 2 of 18 | 0.0017 | 60 | **4.11** |
| ba_n18 | C47 | 0.066 | 497 | 268 / 168 | 30 | 2 of 18 | 0.0017 | 63 | **4.34** |
| ba_n20 | QK3 | 0.174 | 668 | 266 / 171 | 30 | 2 of 20 | 0.0018 | 58 | 5.26 |
| ba_n20 | PSF | 0.141 | 595 | 238 / 150 | 20 | 3 of 21 | 0.0017 | 63 | **6.72** |

"Exposure" is each used qubit's window from its first gate to the end of the circuit divided by its T2, summed
(state-blind; Heron durations 32 ns and 68 ns).

- **Use-weighted averages do not separate the arms:** T1, T2 and CZ error are within a few per cent, and no arm put
  CZ gates on couplers above the device's 80th percentile.
- **Low-T2 qubits do:** in the three BA tests PSF lost (n14, n18, n20), the PSF arm used one more qubit below the
  T2 20th percentile than QK3 (qubit 88, T2 20 us, in n14 and n20; qubit 52, T2 53 us, in n18), and its exposure was
  higher (2.64 against 2.16, 4.11 against 3.27, 6.72 against 5.26). In ba_n16, where PSF was level (1.014), its
  exposure was lower (3.90 against 4.15).
- **Within a test, exposure orders the arms by F** where three arms ran: ba_n16 C47 1.93 < PSF 3.90 < QK3 4.15
  (F 0.297 > 0.284 > 0.280); ba_n18 QK3 3.27 < PSF 4.11 < C47 4.34 (F 0.105 > 0.078 > 0.066).
- **Answer to the question:** PSF did not pick worse couplers or more heavily used couplers; it picked placements
  that keep a low-T2 qubit (qubit 88 at 20 us) in the circuit for its whole length. ESP does not see T2 at all, and
  a14's state-aware idle term sees it but charged ba_n14's PSF arm too little (predicted 1.05, real 0.78).
- **Consistent with the literature note** (`heron-idle-literature-2026-10-13.md`): the published T2 is a Hahn-echo
  time; an unprotected waiting qubit dephases faster than T2 says, and a state-blind window/T2 charge, which assumes
  every qubit is fully sensitive, fitted these four tests better than the state-aware one. This is a hypothesis for
  the next pre-registration, not a finding: four tests, chosen after the fact.

## 5. What follows (proposals; the owner decides)

1. For the replication: pre-register a14's exact estimate (Aer reduced states) as the predictor, with the state-blind
   exposure as a second, separately scored predictor. Fix in advance which tests are left out.
2. Schedule the submitted circuits explicitly (ASAP or ALAP with delays), because the hardware's default timing
   without DD is not documented (literature note, section 4).
3. A placement rule to try offline before any hardware: avoid qubits whose T2 is far below the device's median for
   circuits longer than a few us (the C47 arm of ba_n16 used none and had the best F).
