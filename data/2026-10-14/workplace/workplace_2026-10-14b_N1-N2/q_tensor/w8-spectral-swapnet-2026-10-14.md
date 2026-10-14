# Owner's question (2026-10-14): does a placement from tensor / spectral analysis, or a swap network, beat Qiskit level 3 on circuits that need SWAPs? No.

**Status: offline feasibility check, stopped after it (as home also recommended).** Script `w8_tn_swapnet.py`, rows
`run/w8_rows.json`. Device: ibm_marrakesh's calibration snapshot of 2026-10-13 11:37 as a Target, with W5's cut.

| circuit | QK3 (CZ, us) | spectral order on the snake line | spectral order on a compact region | swap network on the line |
|---|---|---|---|---|
| ba_n14 | 391, 18.4 | 442, 23.1 | 400, 24.9 | 455, 23.3 |
| ba_n16 | 477, 21.0 | 553, 24.4 | 633, 32.1 | 525, 21.8 |
| ba_n18 | 584, 24.3 | 664, 28.6 | 594, 27.6 | 580, 28.7 |
| ba_n20 | 668, 32.8 | 817, 32.0 | 821, 37.3 | 747, 33.9 |
| R4 n=24 (3) | 98-122 CZ | 1.3-2.6x QK3's CZ | 2.2-2.4x | n/a |
| R4 n=48 (3) | 247-324 CZ | 2.7-3.3x | 2.5-2.9x | n/a |

- **Spectral order:** the logical qubits sorted by the Fiedler vector of the weighted interaction graph (a fixed,
  error-free ordering, the one-dimensional case of a contraction tree). Routed by Qiskit level 2 on the cut lattice
  with no Target.
- **Swap network:** odd-even transposition on the line from the spectral order, with each layer stopped once its terms
  are done. Home's version, the full network, was 1.2-1.7x QK3.
- **Correctness:** 25 circuits checked exact. The nine 48-qubit R4 circuits were not checked: the MPS check was too
  slow, and the stabilizer check needs rz angles that are multiples of pi/2. Not one was found wrong.
- **Against c49:** c49 (BA 0.78x QK3, Addendum 477) is further ahead still.
- **Side finding:** Aer's matrix-product-state method returned 0 where the statevector gave 1 for the same amplitude
  (`save_amplitudes_squared`, 24 qubits). The checks here read per-qubit Z expectations instead.
