# SWAPNET (exploratory, nothing predicted)

git head 0b7f790, uncommitted tracked changes: none; 67 HamLib FakeTorino tests on FakeTorino, FakeKingston; 268 jobs.

Not eligible (FakeTorino, SN1): a Z term of weight > 2: 14; a term with X or Y: 47; no path of 100 working qubits found: 2; no path of 112 working qubits found: 1

## FakeTorino

| arm | eligible | exact | ESP = 0 | q2 / QK3 (gmean, +1) | q2 / QK2 | q2 / C32D | ESP / QK3 (gmean, both > 0) | ESP / C32D | wins / losses vs QK3 (ESP, 10%) | time (s, summed) | QK3 time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SN1 | 0 | 0 | 0 | nan | nan | nan | nan | nan | 0 / 0 | 0.0 | 0.0 |
| SNC | 0 | 0 | 0 | nan | nan | nan | nan | nan | 0 / 0 | 0.0 | 0.0 |

| test | n | terms | SNC q2 | QK2 q2 | QK3 q2 | C32D q2 | SNC log10 ESP | QK3 log10 ESP | exact |
|---|---|---|---|---|---|---|---|---|---|
- error (ile[ham_tsp_prob-d198_Ncity-8_enc-unary], SN1): ^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/transpile
- error (ans_transpile[ham_gnp-k_5_n-24_rinst-04], SN1): uli, coeff)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/s
- error (ile[ham_tsp_prob-d198_Ncity-8_enc-unary], SNC): ^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/transpile
- error (e[ham_tsp_prob-lin105_Ncity-7_enc-unary], SNC): ^^^^^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/trans
- error (ans_transpile[ham_gnp-k_5_n-24_rinst-04], SNC): uli, coeff)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/s
- error (e[ham_tsp_prob-lin105_Ncity-7_enc-unary], SN1): ^^^^^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/trans

## FakeKingston

| arm | eligible | exact | ESP = 0 | q2 / QK3 (gmean, +1) | q2 / QK2 | q2 / C32D | ESP / QK3 (gmean, both > 0) | ESP / C32D | wins / losses vs QK3 (ESP, 10%) | time (s, summed) | QK3 time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SN1 | 0 | 0 | 0 | nan | nan | nan | nan | nan | 0 / 0 | 0.0 | 0.0 |
| SNC | 0 | 0 | 0 | nan | nan | nan | nan | nan | 0 / 0 | 0.0 | 0.0 |

| test | n | terms | SNC q2 | QK2 q2 | QK3 q2 | C32D q2 | SNC log10 ESP | QK3 log10 ESP | exact |
|---|---|---|---|---|---|---|---|---|---|
- error (le[ham_tsp_prob-pr76_Ncity-10_enc-unary], SNC): ^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/transpile
- error (e[ham_tsp_prob-lin105_Ncity-7_enc-unary], SNC): ^^^^^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/trans
- error (ile[ham_tsp_prob-d198_Ncity-8_enc-unary], SNC): ^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/transpile
- error (ans_transpile[ham_gnp-k_5_n-24_rinst-04], SN1): uli, coeff)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/s
- error (le[ham_tsp_prob-pr76_Ncity-10_enc-unary], SN1): ^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/transpile
- error (le[ham_tsp_prob-st70_Ncity-10_enc-unary], SNC): ^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/transpile
- error (ans_transpile[ham_gnp-k_5_n-24_rinst-04], SNC): uli, coeff)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/s
- error (le[ham_tsp_prob-st70_Ncity-10_enc-unary], SN1): ^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/transpile
- error (e[ham_tsp_prob-lin105_Ncity-7_enc-unary], SN1): ^^^^^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/trans
- error (ile[ham_tsp_prob-d198_Ncity-8_enc-unary], SN1): ^^^^^^^^^^^^^
  File "/home/<user>/psf_zero_fresh_test/psf_zero_wsl_env_fresh/lib/python3.12/site-packages/qiskit/transpile

