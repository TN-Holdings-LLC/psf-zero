# C25-ID

git head af2e640, uncommitted tracked changes: none; 152 tests (expected 152), 372 jobs, 12 at a time, 14 CPUs, Python 3.11.9
versions as expected in every record: True

- pairs both arms finished: 111 of 186
- identical outputs: 87; different: 24
- failures: REL 55, C25 59; the same set: False
- compile time C25 / REL: median 0.410, geometric mean 0.422
- no pair under 1 s
  - different: test_QASMBench_large[qugan_n39-all-to-all] (default)
  - different: test_QASMBench_large[ghz_n78-square] (default)
  - different: test_hamiltonians[ham_ham_parity-4-all-to-all] (default)
  - different: test_QASMBench_small[inverseqft_n4-linear] (default)
  - different: test_QASMBench_large[bv_n30-square] (default)
  - different: test_QASMBench_large[bv_n30-all-to-all] (default)
  - different: test_circSU2_89_transpile (default)
  - different: test_circSU2_89_transpile (recommended)
  - different: test_QASMBench_large[bv_n140-linear] (default)
  - different: test_QASMBench_small[qec_sm_n5-all-to-all] (default)
  - different: test_hamiltonians[ham_graph-2D-grid-pbc-qubitnodes_Lx-5_Ly-186_h-3-heavy-hex] (default)
  - different: test_QASMBench_small[qaoa_n6-all-to-all] (default)
  - different: test_hamiltonians[ham_mu_x_prime_enc_unary_dvalues_4-4-4-all-to-all] (default)
  - different: test_circSU2_100_transpile (default)
  - different: test_circSU2_100_transpile (recommended)
  - different: test_QASMBench_large[knn_n41-all-to-all] (default)
  - different: test_QASMBench_small[basis_trotter_n4-all-to-all] (default)
  - different: test_QASMBench_medium[qec9xz_n17-all-to-all] (default)
  - different: test_QASMBench_small[vqe_n4-square] (default)
  - different: test_QASMBench_small[vqe_uccsd_n4-heavy-hex] (default)
  - different: test_QASMBench_medium[ghz_state_n23-square] (default)
  - different: test_QASMBench_large[ising_n66-square] (default)
  - different: test_QASMBench_small[cat_state_n4-all-to-all] (default)
  - different: test_QASMBench_medium[seca_n11-all-to-all] (default)
  - failed (REL): test_QAOA_100_transpile (default)
  - failed (REL): test_QASMBench_large[bwt_n37-linear] (default)
  - failed (REL): test_QASMBench_large[knn_129-heavy-hex] (default)
  - failed (REL): test_QASMBench_large[multiplier_n400-heavy-hex] (default)
  - failed (REL): test_QASMBench_large[square_root_n45-all-to-all] (default)
  - failed (REL): test_QASMBench_large[square_root_n45-linear] (default)
  - failed (REL): test_QASMBench_large[square_root_n60-heavy-hex] (default)
  - failed (REL): test_QASMBench_medium[bv_n19-heavy-hex] (default)
  - failed (REL): test_QASMBench_medium[dnn_n16-heavy-hex] (default)
  - failed (REL): test_QASMBench_medium[factor247_n15-linear] (default)
  - failed (C25): test_QAOA_100_transpile (recommended)
  - failed (C25): test_QASMBench_large[32-linear] (default)
  - failed (C25): test_QASMBench_large[bwt_n37-linear] (default)
  - failed (C25): test_QASMBench_large[ising_n420-square] (default)
  - failed (C25): test_QASMBench_large[knn_129-heavy-hex] (default)
  - failed (C25): test_QASMBench_large[qugan_n111-all-to-all] (default)
  - failed (C25): test_QASMBench_large[square_root_n45-all-to-all] (default)
  - failed (C25): test_QASMBench_large[square_root_n45-linear] (default)
  - failed (C25): test_QASMBench_large[square_root_n60-heavy-hex] (default)
  - failed (C25): test_QASMBench_medium[bv_n14-square] (default)

VERDICT NOT IDENTICAL (see above)
