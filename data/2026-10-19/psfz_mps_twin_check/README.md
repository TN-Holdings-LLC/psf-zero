# psfz MPS twin (prototype): check of 2026-10-19 (exploratory; not pre-registered)

Run at home on 2026-10-19, about 14:45 CEST, in the cloud workspace (one CPU core per run), with
`python rust/psfz/harness/twin_check.py OUT` from the repository root (psfz built from this commit). Seed 20261019.
Raw rows: `twin_check.json`.

`psfz twin DEVICE IN OUT [CHI_MAX]` compiles with r4's `compile` (unchanged) and simulates the chain output as a
matrix product state, noiseless. Each cut's bond dimension is bounded before the run by
chi_i <= min(2^(i+1), 2^(n-i-1), 2^(CZ crossing cut i)); a run whose bound exceeds CHI_MAX is refused before it
starts. Splits are exact SVDs (one-sided Jacobi, no outside crate) that drop only singular values below 1e-12 of the
largest; the dropped weight is reported.

TFIM chains (rzz between neighbours, rx fields, u at the end), FakeFez:

| n | steps | bond dimension reached | formula bound | simulate | check |
|---|---|---|---|---|---|
| 8, 12, 16, 20 | 2 / 3 / 4 | 4 / 8 / 16 | 2^4 / 2^4-2^6 / 2^4-2^8 | 0.001-0.14 s | <Z> within 6.5e-15 of Qiskit's state vector |
| 50 | 2 / 4 / 6 | 4 / 16 / 64 | 2^4 / 2^8 / 2^12 | 0.003 / 0.31 / 26.0 s | dropped weight 6.7e-32 / 4.1e-31 / 2.7e-23 |
| 80 | 2 / 4 / 6 | 4 / 16 / 64 | 2^4 / 2^8 / 2^12 | 0.004 / 0.46 / 42.3 s | dropped weight 1.3e-30 / 7.0e-31 / 1.1e-23 |

- **Vanity figure:** an 80-qubit chain, 6 Trotter steps, simulated exactly (dropped weight 1e-23) on one CPU core,
  with every bond dimension bounded before the run.
- **Essential figures:**
  - The bound counts CZ; psfz writes each rzz as 2 CZ, so the bound is 2^(2 steps) where 2^steps is reached. A bound
    read from the input's gates (an rzz has operator Schmidt rank 2) would be tight; not done.
  - The dependency-free Jacobi SVD dominates the time: 26-42 s at bond dimension 64. A LAPACK-class SVD or the GPU is
    needed for real use (a dependency decision for the owner).
  - Chains only (rings refused: the closing edge spans every cut); noiseless; FakeFez's longest working chain for
    psfz is 80-89 qubits (90 and 100 were refused by `compile`).
