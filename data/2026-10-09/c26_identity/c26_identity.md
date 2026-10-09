# C26-ID

git head 9e1cd3a, uncommitted tracked changes: none; 300 cases, 3300 values compared (expected 3300); versions 2026-10-07.1, 2026-10-09.c26; Python 3.11.9

- values that differ: 0

| function | REL (s) | C26 (s) | C26 / REL |
|---|---|---|---|
| excitation_cost | 4.05 | 3.07 | 0.757 |
| hybrid_cost | 4.29 | 3.17 | 0.740 |
| pauli_cost | 4.77 | 4.62 | 0.968 |
| kraus_cost | 6.49 | 6.65 | 1.025 |
| readout_cost | 0.11 | 0.11 | 1.001 |
| _ops_of | 1.57 | 0.66 | 0.420 |
| _implements | 8.10 | 5.48 | 0.677 |
| _same_action | 11.02 | 7.91 | 0.718 |

Recommended call, whole compile (reported only):

- test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_8-8-8]: REL [34.258, 34.783] s, C26 [43.947, 27.464] s; two-qubit gates REL [11688, 11688], C26 [11688, 11688]
- test_feynman_transpile[grover_5.qasm]: REL [1.163, 2.974] s, C26 [0.825, 1.115] s; two-qubit gates REL [537, 537], C26 [537, 537]
- test_hamlib_hamiltonians_transpile[ham_ham_JW-10]: REL [9.951, 7.045] s, C26 [5.583, 5.146] s; two-qubit gates REL [2371, 2371], C26 [2371, 2371]

VERDICT IDENTICAL
