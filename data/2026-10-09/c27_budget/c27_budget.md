# C27-B

git head be9e8e8, uncommitted tracked changes: none; 34 tests (expected 34), 102 jobs, 4 at a time, job timeout 600 s, 14 CPUs, Python 3.11.9

- B0 the run is as locked: **PASS**
- tests where NB and NB2 both finished with the same output: 31 of 34
- B1 where WB refused nothing, WB's output is NB's: 26 of 26: **CONFIRMED**
- B2 WB refuses a call exactly where NB's work exceeds the budget: 31 of 31: **CONFIRMED**
- B3 every WB job finishes (34 of 34), and its estimates and checks take at most 30 s per test (1 over): **REFUTED** (over: test_feynman_transpile[hwb10.qasm] 37.8 s)

## Where the budget binds (reported without prediction)

| test | NB work (s of units) | WB refused | q2 NB / NB2 / WB | compile s NB / NB2 / WB | estimates and checks s NB / WB |
|---|---|---|---|---|---|
| test_feynman_transpile[hwb10.qasm] | x | 5 | x / x / 113292 | x / x / 176.6 | x / 37.8 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | 89.8 | 4 | 3616 / 3616 / 4196 | 145.7 / 142.6 / 34.6 | 135.0 / 22.2 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_8-8- | 57.9 | 4 | 11688 / 11688 / 13261 | 74.8 / 77.7 / 32.4 | 54.1 / 12.3 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-10] | 11.3 | 2 | 2371 / 2371 / 2383 | 24.6 / 25.2 / 17.5 | 18.1 / 12.4 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | 153.9 | 5 | 10856 / 10856 / 12927 | 199.3 / 179.0 / 16.7 | 181.5 / 2.9 |
| test_hamlib_hamiltonians_transpile[ham_ham_parity10] | 11.5 | 2 | 2367 / 2367 / 2404 | 20.2 / 22.0 / 18.7 | 14.5 / 12.9 |

x: the job did not finish (killed or error).

VERDICT SEE ABOVE
