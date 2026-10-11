# GAP-DIAG (a diagnosis, nothing predicted)

git head fc8a5de; 106 tests on FakeTorino, FakeKingston, FakeMarrakesh, FakeFez; 848 jobs; errors 0.

Parts are sums of -log10(1 - error); log10 ESP = -(two-qubit + one-qubit + measurement). Means over the runnable tests (best ESP of the two >= 0.01) where neither output uses a failed element; a positive difference is REL's extra loss.

| device | tests | ESP REL / QK3 (gmean) | two-qubit part: REL - QK3 | ... from the count | ... from the error per gate | one-qubit part | measurement part | two-qubit gates REL / QK3 | mean error per two-qubit gate REL / QK3 | qubits used REL / QK3 | REL worse by 10% / better by 10% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FakeTorino | 53 | 0.981 | +0.00173 | -0.00160 | +0.00334 | +0.00125 | +0.00517 | 10013 / 10071 | 0.00157 / 0.00155 | 8 / 8 | 5 / 5 |
| FakeKingston | 64 | 1.017 | -0.01350 | -0.00230 | -0.01119 | +0.00348 | +0.00271 | 27059 / 27256 | 0.00070 / 0.00073 | 10.0 / 10.0 | 6 / 8 |
| FakeMarrakesh | 56 | 0.994 | -0.00099 | +0.00693 | -0.00793 | +0.00330 | +0.00016 | 14888 / 14540 | 0.00095 / 0.00098 | 8.0 / 8.0 | 6 / 5 |
| FakeFez | 55 | 0.985 | +0.00336 | +0.00800 | -0.00464 | +0.00228 | +0.00099 | 12610 / 12319 | 0.00138 / 0.00140 | 8 / 8 | 7 / 3 |

Per test (all devices), the 15 where REL loses most:

| device | test | log10 ESP REL - QK3 | two-qubit REL / QK3 | two-qubit part | one-qubit part | measurement part |
|---|---|---|---|---|---|---|
| Marrakesh | gf2^7_mult.qasm | -0.3830 | 803 / 708 | +0.3094 | +0.0736 | +0.0000 |
| Fez | gf2^7_mult.qasm | -0.1805 | 866 / 726 | +0.1433 | +0.0372 | +0.0000 |
| Torino | ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-1 | -0.1781 | 1058 / 1010 | +0.1470 | +0.0311 | +0.0000 |
| Fez | ham15-low.qasm | -0.1767 | 787 / 785 | +0.1577 | +0.0190 | +0.0000 |
| Marrakesh | ham15-med.qasm | -0.1562 | 1238 / 1202 | +0.1098 | +0.0463 | +0.0000 |
| Marrakesh | ham15-low.qasm | -0.1221 | 840 / 744 | +0.0894 | +0.0326 | +0.0000 |
| Torino | adder_8.qasm | -0.1177 | 746 / 730 | +0.0893 | +0.0284 | +0.0000 |
| Torino | inverseqft1.qasm | -0.1076 | 0 / 0 | +0.0000 | +0.0000 | +0.1076 |
| Torino | inverseqft2.qasm | -0.1076 | 0 / 0 | +0.0000 | +0.0000 | +0.1076 |
| Fez | qcla_adder_10.qasm | -0.1011 | 526 / 505 | +0.0921 | +0.0090 | +0.0000 |
| Marrakesh | tof_10.qasm | -0.0924 | 188 / 145 | +0.0739 | +0.0185 | +0.0000 |
| Marrakesh | ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-1 | -0.0910 | 1106 / 998 | +0.0773 | +0.0137 | +0.0000 |
| Marrakesh | ham_ham_JW-8 | -0.0860 | 1273 / 1345 | +0.0707 | +0.0153 | +0.0000 |
| Kingston | inverseqft2.qasm | -0.0846 | 0 / 0 | +0.0000 | -0.0000 | +0.0846 |
| Kingston | inverseqft1.qasm | -0.0846 | 0 / 0 | +0.0000 | -0.0000 | +0.0846 |
